import torch
import torch.nn as nn
from modules.box_attention import BoxAttention
from modules.feed_forward import FeedForward
from timm.layers import DropPath


class Sboxblock(nn.Module):
    """
    A single Swin Transformer-like block for 3D voxel data, incorporating
    window-based multi-head self-attention and a Feed-Forward Network (FFN).
    It supports optional shifting of windows for cross-window interaction.
    """
    def __init__(self, out_channels, resolution, boxsize, mlp_dims, shift=True, drop_path=0.):
        """
        Initializes the Sboxblock.

        Args:
            out_channels (int): The feature dimension (C) of the input/output.
            resolution (int): The resolution (R) of the 3D voxel grid.
            boxsize (int): The size of the cubic attention window.
            mlp_dims (int): The hidden dimension for the FeedForward network.
            shift (bool): If True, enables shifted window attention. If None, no shifting.
            drop_path (float): Dropout rate for DropPath (stochastic depth).
        """
        super().__init__()
        self.out_channels = out_channels
        self.resolution = resolution
        self.heads = 4 # Fixed number of attention heads for this block
        self.dim_head = self.out_channels // self.heads # Dimension per head
        self.box_size = boxsize

        # Determine shift size for shifted window attention.
        # If 'shift' is True, shift by half the box size. Otherwise, no shift.
        if shift is not None:
            self.shift_size = self.box_size // 2
        else:
            self.shift_size = 0

        # Initialize the BoxAttention module.
        self.attn = BoxAttention(
            dim=self.out_channels,
            box_size=self.box_size,
            num_heads=self.heads
        )

        # --- Create attention mask for shifted windows ---
        # This mask is crucial for shifted window attention. When windows are shifted,
        # some elements within a "shifted window" might logically belong to different
        # original (non-shifted) windows. The mask prevents attention computation
        # between these logically distinct regions, maintaining local attention.
        if self.shift_size > 0:
            # Create a 3D mask grid, initialized to zeros.
            # Shape: (1, R, R, R, 1) - batch, spatial dims, dummy channel.
            img_mask = torch.zeros((1, self.resolution, self.resolution, self.resolution, 1))
            # Define slices for partitioning the grid into 3x3x3 regions (for 3D).
            # These slices represent the different "quadrants" or "octants" that
            # result from shifting.
            slices = (slice(0, -self.box_size), # Region before the last full box
                      slice(-self.box_size, -self.shift_size), # Region covering part of the last box and shift
                      slice(-self.shift_size, None)) # Region covering the shifted part

            cnt = 0 # Counter for assigning unique IDs to different mask regions.
            # Iterate through all combinations of slices to assign unique IDs to regions.
            # This creates a pattern where different regions have different integer IDs.
            for x_slice in slices:
                for y_slice in slices:
                    for z_slice in slices:
                        img_mask[:, x_slice, y_slice, z_slice, :] = cnt
                        cnt += 1

            # Partition the mask grid into boxes, similar to how features are partitioned.
            mask_boxs = box_partition(img_mask, self.box_size)
            # Flatten each mask box into a 1D sequence of IDs. Shape: (-1, box_size^3)
            mask_boxs = mask_boxs.view(-1, self.box_size ** 3)

            # Compute the attention mask.
            # For each pair of elements (i, j) within a flattened box, if their
            # mask IDs are different (mask_boxs[k, i] != mask_boxs[k, j]), it means
            # they belong to different logical windows after shifting.
            # In such cases, the attention score between them is set to a very small
            # negative number (float(-100.0)), which becomes effectively zero after softmax.
            # If mask IDs are the same, the attention score is set to 0.0 (no masking).
            attn_mask = mask_boxs.unsqueeze(1) - mask_boxs.unsqueeze(2) # Shape: (-1, box_size^3, box_size^3)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            # If no shifting, no attention mask is needed.
            attn_mask = None

        # Register the attention mask as a buffer.
        self.register_buffer("attn_mask", attn_mask)

        # Layer Normalization layers. Applied before attention and MLP.
        self.norm1 = nn.LayerNorm(out_channels) # For input to attention
        self.norm2 = nn.LayerNorm(out_channels) # For input to MLP

        # FeedForward Network (MLP).
        self.mlp_dim = mlp_dims
        self.mlp = FeedForward(out_channels, self.mlp_dim)

        # DropPath (Stochastic Depth): A regularization technique where paths
        # in the network are randomly dropped during training.
        self.drop_path = DropPath(drop_path)

    def forward(self, inputs):
        """
        Performs the forward pass of the Sboxblock.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, R^3, C_out),
                                   where R^3 is the flattened resolution.

        Returns:
            torch.Tensor: Output tensor of the same shape (B, R^3, C_out).
        """
        shortcut = inputs # Store input for residual connection
        b = inputs.shape[0] # Batch size

        # Apply LayerNorm to the input.
        x = self.norm1(inputs)
        # Reshape from flattened (B, R^3, C) to 3D (B, R, R, R, C) for spatial operations.
        x = torch.reshape(x, (b, self.resolution, self.resolution, self.resolution, self.out_channels))

        # Perform cyclic shifting if `shift_size` is greater than 0.
        # This shifts the feature map by `shift_size` voxels in negative x, y, z directions.
        # This enables interaction between different windows in alternating blocks.
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))
        else:
            # If no shifting, just use the original (reshaped) tensor.
            shifted_x = x

        # Partition the (shifted) feature map into non-overlapping boxes.
        # Shape: (B * (R/box_size)^3, box_size, box_size, box_size, C_out)
        boxs = box_partition(shifted_x, self.box_size)
        # Reshape each box to a flattened sequence for attention.
        # Shape: (-1, box_size^3, C_out)
        boxs = torch.reshape(boxs, (-1, self.box_size ** 3, self.out_channels))

        # Apply BoxAttention within each box.
        # The attention mask is passed if shifting is enabled.
        attn_boxs = self.attn(boxs, mask=self.attn_mask)
        # Output shape: (-1, box_size^3, C_out)

        # Reshape the attended boxes back to their 3D window format.
        # Shape: (-1, box_size, box_size, box_size, C_out)
        boxs = torch.reshape(attn_boxs, (-1, self.box_size, self.box_size, self.box_size,
                                               self.out_channels))
        # Reverse the box partitioning to reconstruct the (shifted) 3D feature map.
        # Shape: (B, R, R, R, C_out)
        x = box_reverse(boxs, self.box_size, self.resolution)

        # Reverse the cyclic shifting if it was applied.
        # This brings the feature map back to its original alignment.
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size),
                                   dims=(1, 2, 3))
        else:
            shifted_x = x
        # Reshape the 3D feature map back to flattened (B, R^3, C) for subsequent operations.
        x = torch.reshape(shifted_x, (b, self.resolution**3, self.out_channels))

        # First residual connection and DropPath.
        # Applies DropPath to the attention output, scales by 0.5 (common in some architectures),
        # and adds it to the original shortcut (input).
        x = self.drop_path(x) * 0.5 + shortcut

        # Second residual connection and DropPath, after MLP.
        # Apply LayerNorm, then MLP, then DropPath, scale by 0.5, and add to 'x'.
        x = self.drop_path(self.mlp(self.norm2(x))) * 0.5 + x

        return x

def box_partition(x, box_size):
    """
    Partitions a 3D feature map into non-overlapping 3D "boxes" (windows).
    This is a crucial step for implementing window-based attention,
    similar to Swin Transformers, where attention is computed locally within
    these smaller windows rather than globally over the entire feature map.

    Args:
        x (torch.Tensor): Input tensor of shape (B, R, R, R, C_out),
                          where B is batch size, R is resolution, and C_out is channels.
        box_size (int): The size of the cubic window (e.g., 3 for a 3x3x3 window).

    Returns:
        torch.Tensor: A tensor of shape (-1, box_size, box_size, box_size, C_out),
                      where -1 indicates the total number of boxes (B * (R/box_size)^3).
                      Each element in the first dimension is a flattened 3D window.
    """
    # Extract batch size, resolution, and output channels from the input tensor.
    b = x.shape[0]
    resolution = x.shape[1] # Assuming resolution is the same for all spatial dims
    out_channels = x.shape[-1]

    # Reshape the input tensor to explicitly define the windows.
    # Example: (B, R, R, R, C) -> (B, R/box_size, box_size, R/box_size, box_size, R/box_size, box_size, C)
    # This creates dimensions for the "number of windows" along each axis
    # and the "elements within each window" along each axis.
    x = torch.reshape(x, (
        b,
        resolution // box_size, box_size, # Z-windows, Z-elements-in-window
        resolution // box_size, box_size, # Y-windows, Y-elements-in-window
        resolution // box_size, box_size, # X-windows, X-elements-in-window
        out_channels))

    # Permute and reshape to group elements by window.
    # The permute operation rearranges dimensions so that the window indices (1, 3, 5)
    # come before the in-window indices (2, 4, 6).
    # Then, .contiguous().view() flattens the batch and window dimensions into one,
    # resulting in a tensor where each row is a flattened window.
    boxs = x.permute(0, 1, 3, 5, # Batch index, then window indices (Z_win, Y_win, X_win)
                     2, 4, 6,    # In-window element indices (Z_elem, Y_elem, X_elem)
                     7).contiguous().view(-1, box_size, box_size, box_size, out_channels)
    # Resulting shape: (B * (R/box_size)^3, box_size, box_size, box_size, C_out)

    return boxs

def box_reverse(boxs, box_size, resolution):
    """
    Reverses the `box_partition` operation, reconstructing the original 3D feature map
    from a collection of flattened 3D boxes (windows).

    Args:
        boxs (torch.Tensor): Input tensor of shape (-1, box_size, box_size, box_size, C_out),
                             where -1 is the total number of boxes.
        box_size (int): The size of the cubic window.
        resolution (int): The original resolution (R) of the 3D feature map.

    Returns:
        torch.Tensor: The reconstructed 3D feature map of shape (B, R, R, R, C_out).
    """
    # Calculate the original batch size (B).
    # The total number of boxes is (B * (R/box_size)^3).
    # So, B = total_boxes / ((R/box_size)^3).
    b = int(boxs.shape[0] / (resolution ** 3 / box_size / box_size / box_size))
    # Note: resolution ** 3 / box_size / box_size / box_size simplifies to (resolution / box_size)**3

    # Reshape the flattened boxes back into a structure that reflects the
    # original arrangement of windows within the 3D grid.
    # This is the inverse of the first reshape in `box_partition`.
    x = torch.reshape(boxs, (
        b,
        resolution // box_size, # Number of windows along Z
        resolution // box_size, # Number of windows along Y
        resolution // box_size, # Number of windows along X
        box_size, box_size, box_size, # Dimensions of each window
        -1)) # Channels (C_out)

    # Permute and reshape to reconstruct the original 3D feature map.
    # This is the inverse of the permute and view in `box_partition`.
    # It rearranges dimensions from (B, Z_win, Y_win, X_win, Z_elem, Y_elem, X_elem, C)
    # back to (B, Z_total, Y_total, X_total, C).
    x = x.permute(0, # Batch
                  1, 4, # Z_win, Z_elem -> combines to Z_total
                  2, 5, # Y_win, Y_elem -> combines to Y_total
                  3, 6, # X_win, X_elem -> combines to X_total
                  7).contiguous().view(b, resolution, resolution, resolution, -1)
    # Resulting shape: (B, R, R, R, C_out)
    return x
