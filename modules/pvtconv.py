import torch
import torch.nn as nn
# Assuming these are custom modules defined elsewhere, providing their functionality.
# F: Likely a functional module for operations like trilinear_devoxelize.
# Voxelization: Module to convert point clouds to voxel grids.
# SharedTransformer: A transformer module for point features.
# SE3d: A 3D Squeeze-and-Excitation module for channel-wise re-weighting.
import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_transformer import SharedTransformer
from modules.se import SE3d
from timm.models.layers import DropPath # Used for stochastic depth
import numpy as np

# Defines the public API of this module, specifying which classes are exposed
# when `from . import module_name` is used.
__all__ = ['PVTConv','PartPVTConv','SemPVTConv']

def rand_bbox(size, lam):
    """
    Generates a random bounding box for CutMix/Cutout data augmentation in 3D.
    CutMix is a regularization technique that mixes two training examples and their labels.
    Cutout is a similar technique where a random rectangular region of an image is masked out.
    This function adapts it for 3D voxel grids.

    Args:
        size (tuple): The shape of the input tensor, typically (B, C, D, H, W) for 3D data.
                      Here, it's used as (B, C, Z, Y, X) where Z, Y, X are spatial dimensions.
        lam (float):  The mixing coefficient (lambda) from the Beta distribution,
                      determining the size of the cutout region. A value close to 1
                      means a small cutout, close to 0 means a large cutout.

    Returns:
        tuple: (bbx1, bby1, bbx2, bby2, bbz1, bbz2)
               The coordinates defining the bounding box (min_z, min_y, min_x, max_z, max_y, max_x).
               These are used to select a region to mix or cut.
    """
    # Extract spatial dimensions (Z, Y, X) from the input tensor's shape.
    x = size[2] # Corresponds to Z (depth)
    y = size[3] # Corresponds to Y (height)
    z = size[4] # Corresponds to X (width)

    # Calculate the size of the cutout region based on lambda.
    # cut_rat is the ratio of the cutout side length to the original side length.
    cut_rat = np.sqrt(1. - lam)
    # Calculate the absolute dimensions of the cutout box.
    cut_x = np.int(x * cut_rat) # Cutout depth
    cut_y = np.int(y * cut_rat) # Cutout height
    cut_z = np.int(z * cut_rat) # Cutout width

    # Randomly choose the center coordinates (cx, cy, cz) for the cutout box.
    cx = np.random.randint(x) # Center depth
    cy = np.random.randint(y) # Center height
    cz = np.random.randint(z) # Center width

    # Calculate the bounding box coordinates (min and max for each dimension).
    # np.clip ensures the coordinates stay within the valid range [0, spatial_dim].
    bbx1 = np.clip(cx - cut_x // 2, 0, x) # Min depth
    bbx2 = np.clip(cx + cut_x // 2, 0, x) # Max depth
    bby1 = np.clip(cy - cut_y // 2, 0, y) # Min height
    bby2 = np.clip(cy + cut_y // 2, 0, y) # Max height
    bbz1 = np.clip(cz - cut_z // 2, 0, z) # Min width
    bbz2 = np.clip(cz + cut_z // 2, 0, z) # Max width

    return bbx1, bby1, bbx2, bby2, bbz1, bbz2

class FeedForward(nn.Module):
    """
    A standard Feed-Forward Network (FFN) block, also known as a Multi-Layer Perceptron (MLP).
    This block is typically used within Transformer architectures after the attention mechanism.
    It applies two linear transformations with a GELU activation and dropout in between.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        Initializes the FeedForward module.

        Args:
            dim (int): The input and output feature dimension.
            hidden_dim (int): The dimension of the hidden layer. Typically, hidden_dim > dim
                              to allow the network to learn more complex transformations.
            dropout (float): The dropout rate applied after the GELU activation and
                             after the second linear layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            # First linear transformation: maps input 'dim' to 'hidden_dim'.
            nn.Linear(dim, hidden_dim),
            # Gaussian Error Linear Unit (GELU) activation function.
            # It's a smooth approximation of the ReLU function, often used in Transformers.
            nn.GELU(),
            # Dropout layer: randomly sets a fraction of input units to zero at each update
            # during training time, which helps prevent overfitting.
            nn.Dropout(dropout),
            # Second linear transformation: maps 'hidden_dim' back to 'dim'.
            nn.Linear(hidden_dim, dim),
            # Another dropout layer for regularization.
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Performs the forward pass of the FeedForward block.

        Args:
            x (torch.Tensor): Input tensor, typically of shape (B, N, dim)
                              where N is the sequence length (e.g., number of voxels).

        Returns:
            torch.Tensor: Output tensor of the same shape (B, N, dim).
        """
        return self.net(x)

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

class BoxAttention(nn.Module):
    """
    Implements a 3D window-based multi-head self-attention mechanism,
    similar to the attention block in Swin Transformers, but adapted for 3D.
    It incorporates relative position bias for better spatial understanding.
    """
    def __init__(self, dim, box_size, num_heads, attn_drop=0., proj_drop=0.):
        """
        Initializes the BoxAttention module.

        Args:
            dim (int): The input feature dimension (C_out from previous layers).
            box_size (int): The size of the cubic window (e.g., 3x3x3).
            num_heads (int): The number of attention heads. 'dim' must be divisible by 'num_heads'.
            attn_drop (float): Dropout rate for attention weights.
            proj_drop (float): Dropout rate for the output projection.
        """
        super().__init__()
        self.dim = dim
        self.box_size = box_size
        self.num_heads = num_heads
        # Calculate the dimension of each attention head.
        head_dim = dim // num_heads
        # Scaling factor for attention scores (1/sqrt(d_k)).
        self.scale = head_dim ** -0.5

        # Relative position bias table: a learnable parameter that adds a bias
        # to attention scores based on the relative spatial positions of tokens
        # within a window. This helps the model incorporate spatial inductive bias.
        # For a 3D window of size `box_size`, the maximum relative displacement
        # along one axis is `box_size - 1`. So, the range of relative positions
        # is `-(box_size - 1)` to `(box_size - 1)`, which is `2 * box_size - 1` unique values.
        # For 3D, the sum of relative coordinates is used, so the max possible sum is
        # (box_size-1) + (box_size-1) + (box_size-1) = 3*box_size - 3.
        # The total range is (3*box_size - 1) * (3*box_size - 1) if using 2D indexing,
        # or (3 * box_size - 1) ** 3 if using 3D indexing.
        # The current implementation uses a sum of relative coordinates, so the range is
        # (3 * box_size - 1) for a single axis, and then sum over 3 axes.
        # The size (3 * box_size - 1) ** 2 seems to imply a 2D relative position encoding
        # that is then summed, or a flattened 3D indexing. Let's look at the index calculation.
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((3 * box_size - 1) ** 2, num_heads)) # Shape: (relative_pos_index_range, num_heads)

        # --- Calculate relative position indices ---
        # Create 1D coordinate tensors for each axis within a single box.
        coords_x = torch.arange(self.box_size)
        coords_y = torch.arange(self.box_size)
        coords_z = torch.arange(self.box_size)

        # Create a 3D grid of coordinates (Z, Y, X) for all positions within a box.
        # torch.meshgrid returns tensors for each dimension. torch.stack combines them.
        coords = torch.stack(torch.meshgrid([coords_x, coords_y, coords_z], indexing='ij')) # Shape: (3, box_size, box_size, box_size)

        # Flatten the spatial dimensions of the coordinates.
        # Resulting shape: (3, box_size^3). Each column is (x,y,z) for a voxel.
        coords_flatten = torch.flatten(coords, 1)

        # Compute relative coordinates between all pairs of flattened voxel positions.
        # coords_flatten[:, :, None] is (3, box_size^3, 1)
        # coords_flatten[:, None, :] is (3, 1, box_size^3)
        # Subtraction results in (3, box_size^3, box_size^3).
        # This tensor contains (dx, dy, dz) for every possible pair of voxels in a window.
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]

        # Permute to (box_size^3, box_size^3, 3).
        # This means for each pair (i, j) of voxels, we have (dx_ij, dy_ij, dz_ij).
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()

        # Shift the relative coordinates to be non-negative.
        # This is done to map the relative coordinates to valid indices in the
        # `relative_position_bias_table`.
        # For a box_size of S, relative coords range from -(S-1) to (S-1).
        # Adding (S-1) shifts them to 0 to (2S-2).
        relative_coords[:, :, 0] += self.box_size - 1 # Shift x-coords
        relative_coords[:, :, 1] += self.box_size - 1 # Shift y-coords
        relative_coords[:, :, 2] += self.box_size - 1 # Shift z-coords

        # Further scale and sum the shifted relative coordinates to get a single index.
        # This mapping converts a 3D relative position (dx, dy, dz) into a unique
        # 1D index for the `relative_position_bias_table`.
        # The formula (z_shifted * (max_y_shifted_val * max_x_shifted_val) + y_shifted * max_x_shifted_val + x_shifted)
        # is used to create a unique index.
        # Here, it's simplified by multiplying x by (3*box_size-1) and then summing.
        # This implies a flattened 3D indexing scheme where the first dimension is
        # multiplied by the range of the second, etc.
        # (3 * box_size - 1) is the maximum possible value for a single shifted coordinate.
        relative_coords[:, :, 0] *= (3 * self.box_size - 1) # Scale x_shifted
        # The sum over the last dimension (3) creates a single 1D index for each pair.
        # This is a specific way to flatten 3D relative coordinates into a 1D index.
        relative_position_index = relative_coords.sum(-1) # Shape: (box_size^3, box_size^3)

        # Register `relative_position_index` as a buffer.
        # Buffers are tensors that are not considered model parameters but are part of the model's state.
        # They are saved in the state_dict but not updated by optimizers.
        self.register_buffer("relative_position_index", relative_position_index)

        # Linear layer to project input features into Query (Q), Key (K), and Value (V) matrices.
        # It takes 'dim' input features and outputs 'dim * 3' features, which are then split.
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        # Dropout layer for attention weights.
        self.attn_drop = nn.Dropout(attn_drop)
        # Output linear projection layer.
        self.proj = nn.Linear(dim, dim)
        # Dropout layer for the output projection.
        self.proj_drop = nn.Dropout(proj_drop)

        # Softmax activation applied along the last dimension to normalize attention scores.
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Performs the forward pass of the BoxAttention block.

        Args:
            x (torch.Tensor): Input tensor of shape (B_, N, C), where
                              B_ is the total number of flattened windows across all batches,
                              N is the number of elements (voxels) in a single window (box_size^3),
                              C is the feature dimension.
            mask (torch.Tensor, optional): An attention mask of shape (nW, N, N) where nW is
                                           the number of windows in a batch. Used for shifted
                                           window attention to mask out connections between
                                           different logical windows. Defaults to None.

        Returns:
            torch.Tensor: Output tensor of shape (B_, N, C), representing the attended features.
        """
        # Unpack dimensions from the input tensor.
        B_, N, C = x.shape # B_ is total number of boxes, N is box_size^3, C is dim

        # Project input features 'x' into Q, K, V.
        # self.qkv(x) has shape (B_, N, C*3).
        # .reshape(...) splits C*3 into 3 (for Q, K, V), num_heads, and head_dim.
        # .permute(...) rearranges dimensions to (3, B_, num_heads, N, head_dim)
        # for easier splitting into q, k, v.
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # Split qkv into Q, K, V tensors. Each has shape (B_, num_heads, N, head_dim).
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scale Query (Q) by 1/sqrt(head_dim) to prevent dot products from becoming too large.
        q = q * self.scale

        # Compute attention scores: Q @ K^T.
        # (B_, num_heads, N, head_dim) @ (B_, num_heads, head_dim, N) -> (B_, num_heads, N, N)
        # This matrix contains raw attention scores for each pair of tokens within each head.
        attn = (q @ k.transpose(-2, -1))

        # Apply relative position bias to the attention scores.
        # self.relative_position_index.view(-1) flattens the (box_size^3, box_size^3) index tensor.
        # This flattened index is used to lookup values from `relative_position_bias_table`.
        # The result is then reshaped back to (box_size^3, box_size^3, num_heads).
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.box_size ** 3, self.box_size ** 3, -1)
        # Permute to (num_heads, box_size^3, box_size^3) to align with 'attn' tensor.
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        # Add the bias to the attention scores.
        # .unsqueeze(0) adds a batch dimension (1, num_heads, N, N) to broadcast across batches.
        attn = attn + relative_position_bias.unsqueeze(0)

        # Apply attention mask if provided (for shifted window attention).
        if mask is not None:
            # nW: number of windows in a single batch (B_ / nW is the original batch size)
            nW = mask.shape[0] # Mask shape is (nW, N, N)
            # Reshape 'attn' to expose the original batch dimension for mask application.
            # (B_, num_heads, N, N) -> (B_//nW, nW, num_heads, N, N)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            # Add the mask. mask.unsqueeze(1).unsqueeze(0) makes it (1, nW, 1, N, N)
            # to broadcast across batch and head dimensions.
            # The mask typically contains large negative values (-100.0) for masked connections,
            # which become ~0 after softmax, effectively preventing attention.
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            # Reshape back to (B_, num_heads, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            # Apply softmax to get attention probabilities.
            attn = self.softmax(attn)
        else:
            # If no mask, just apply softmax directly.
            attn = self.softmax(attn)

        # Apply dropout to the attention weights.
        attn = self.attn_drop(attn)

        # Compute the weighted sum of Value (V) features.
        # (B_, num_heads, N, N) @ (B_, num_heads, N, head_dim) -> (B_, num_heads, N, head_dim)
        x = (attn @ v)
        # Transpose to (B_, N, num_heads, head_dim) to prepare for merging heads.
        x = x.transpose(1, 2)
        # Reshape to (B_, N, C) by concatenating head outputs.
        x = x.reshape(B_, N, C)

        # Apply the output projection layer.
        x = self.proj(x)
        # Apply dropout to the projected output.
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        """Provides additional representation for the module, useful for debugging."""
        return f'dim={self.dim}, box_size={self.box_size}, num_heads={self.num_heads}'

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
            out_channels, box_size=self.box_size, num_heads=self.heads)

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

class Transformer(nn.Module):
    """
    A stack of Sboxblock modules, forming a complete Transformer encoder layer
    for 3D voxel features. It typically alternates between non-shifted and
    shifted window attention blocks.
    """
    def __init__(self, out_channels, resolution, boxsize, mlp_dims, drop_path1, drop_path2):
        """
        Initializes the Transformer module.

        Args:
            out_channels (int): Feature dimension.
            resolution (int): Voxel grid resolution.
            boxsize (int): Attention window size.
            mlp_dims (int): Hidden dimension for MLPs in Sboxblock.
            drop_path1 (float): DropPath rate for even-indexed Sboxblocks (no shift).
            drop_path2 (float): DropPath rate for odd-indexed Sboxblocks (shifted).
        """
        super().__init__()
        self.shift = None # Placeholder, actual shift controlled by Sboxblock
        self.depth = 2 # Number of Sboxblock layers in this Transformer module

        # Create a ModuleList of Sboxblock instances.
        # It alternates between non-shifted (shift=None) and shifted (shift=True)
        # attention blocks, and applies different drop_path rates.
        self.blocks = nn.ModuleList([
            Sboxblock(out_channels, resolution, boxsize, mlp_dims,
                      # Shift is None for even blocks (i%2 == 0), True for odd blocks.
                      shift=None if (i % 2 == 0) else True,
                      # DropPath rate alternates between drop_path1 and drop_path2.
                      drop_path=drop_path1 if (i % 2 == 0) else drop_path2)
            for i in range(self.depth)])

    def forward(self, x):
        """
        Performs the forward pass through the stack of Sboxblocks.

        Args:
            x (torch.Tensor): Input tensor of shape (B, R^3, C).

        Returns:
            torch.Tensor: Output tensor of the same shape (B, R^3, C).
        """
        # Iterate through each Sboxblock and apply it sequentially.
        for blk in self.blocks:
            x = blk(x)
        return x

class VoxelEncoder(nn.Module):
    """
    Encodes 3D voxel features. It typically takes a raw voxel grid,
    applies an initial convolution for feature extraction, adds positional
    embeddings, and then processes them through a Transformer block.
    """
    def __init__(self, in_channels, out_channels, kernel_size, resolution, boxsize, mlp_dims, drop_path1, drop_path2):
        """
        Initializes the VoxelEncoder.

        Args:
            in_channels (int): Number of input channels to the voxel grid (e.g., from Voxelization).
            out_channels (int): Number of output channels after encoding.
            kernel_size (int): Kernel size for the initial 3D convolution.
            resolution (int): Voxel grid resolution.
            boxsize (int): Attention window size for the Transformer.
            mlp_dims (int): Hidden dimension for MLPs in Transformer blocks.
            drop_path1 (float): DropPath rate for Transformer blocks.
            drop_path2 (float): DropPath rate for Transformer blocks.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.kernel_size = kernel_size

        # Initial 3D convolutional layer to extract features from the voxel grid.
        # stride=1, padding=kernel_size//2 ensures the output spatial dimensions
        # remain the same as input spatial dimensions (R, R, R).
        self.voxel_emb = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)

        # Layer Normalization applied to the flattened voxel features.
        self.layer_norm = nn.LayerNorm(out_channels)

        # Dropout layer for positional embeddings.
        self.pos_drop = nn.Dropout(p=0.) # Dropout rate is 0.0 by default, effectively disabled.

        # Learnable positional embedding.
        # This tensor adds positional information to each voxel, as Transformers
        # are permutation-invariant without such embeddings.
        # Shape: (1, R^3, C_out) to broadcast across batch dimension.
        self.pos_embedding = nn.Parameter(torch.randn(1, self.resolution ** 3, self.out_channels))

        # Transformer module to process the voxel features.
        self.voxel_Trasformer = Transformer(out_channels, resolution, boxsize, mlp_dims, drop_path1, drop_path2)

    def forward(self, inputs):
        """
        Performs the forward pass of the VoxelEncoder.

        Args:
            inputs (torch.Tensor): Input voxel grid of shape (B, C_in, R, R, R).

        Returns:
            torch.Tensor: Encoded voxel features of shape (B, C_out, R, R, R).
        """
        # Apply the initial 3D convolution.
        inputs = self.voxel_emb(inputs) # Shape: (B, C_out, R, R, R)

        # Reshape the 3D voxel features to flattened (B, C_out, R^3) for LayerNorm and Transformer.
        inputs = torch.reshape(inputs, (-1, self.out_channels, self.resolution ** 3))
        # Permute to (B, R^3, C_out) to match the expected input format for LayerNorm and Transformer.
        x = inputs.permute(0, 2, 1) # Shape: (B, R^3, C_out)

        # Apply Layer Normalization.
        x = self.layer_norm(x)

        # Add the learnable positional embedding.
        # This broadcasts the (1, R^3, C_out) pos_embedding to each item in the batch.
        x += self.pos_embedding  # todo: Comment indicating potential future work or reminder.

        # Apply dropout to the features after adding positional embeddings.
        x = self.pos_drop(x)

        # Pass the features through the Transformer.
        x = self.voxel_Trasformer(x) # Shape: (B, R^3, C_out)

        # Reshape the output from the Transformer back to 3D voxel grid format.
        x = torch.reshape(x, (-1, self.resolution, self.resolution, self.resolution, self.out_channels))
        # Permute dimensions to (B, C_out, R, R, R) to match standard convolutional tensor format.
        x = x.permute(0, 4, 1, 2, 3)
        return x

class SegVoxelEncoder(nn.Module):
    """
    A specialized VoxelEncoder for segmentation tasks, incorporating a CutMix-like
    data augmentation strategy during the forward pass. This helps in regularization
    and improving generalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, resolution, boxsize, mlp_dims, drop_path1, drop_path2):
        """
        Initializes the SegVoxelEncoder. Parameters are similar to VoxelEncoder.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for initial convolution.
            resolution (int): Voxel grid resolution.
            boxsize (int): Attention window size.
            mlp_dims (int): Hidden dimension for MLPs.
            drop_path1 (float): DropPath rate.
            drop_path2 (float): DropPath rate.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.kernel_size = kernel_size
        # Initial 3D convolutional layer.
        self.voxel_emb = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        # Layer Normalization.
        self.layer_norm = nn.LayerNorm(out_channels)
        # Positional embedding dropout.
        self.pos_drop = nn.Dropout(p=0.)
        # Learnable positional embedding.
        self.pos_embedding = nn.Parameter(torch.randn(1, self.resolution ** 3, self.out_channels))
        # Transformer module.
        self.voxel_Trasformer = Transformer(out_channels, resolution, boxsize, mlp_dims, drop_path1, drop_path2)
        # Beta parameter for the Beta distribution used in rand_bbox for CutMix.
        self.beta = 1.

    def forward(self, inputs):
        """
        Performs the forward pass of the SegVoxelEncoder, including CutMix augmentation.

        Args:
            inputs (torch.Tensor): Input voxel grid of shape (B, C_in, R, R, R).

        Returns:
            torch.Tensor: Encoded voxel features of shape (B, C_out, R, R, R),
                          potentially with CutMix applied.
        """
        # Apply initial 3D convolution.
        inputs = self.voxel_emb(inputs) # Shape: (B, C_out, R, R, R)

        # --- CutMix Data Augmentation ---
        # Generate a lambda value from a Beta distribution.
        lam = np.random.beta(self.beta, self.beta)
        # Generate random bounding box coordinates for the cutout region.
        bbx1, bby1, bbx2, bby2, bbz1, bbz2 = rand_bbox(inputs.size(), lam)

        # Create a clone of the input tensor to perform CutMix.
        temp_x = inputs.clone()
        # Apply CutMix: The region defined by the bounding box in the current batch
        # is replaced with the corresponding region from a flipped version of the batch.
        # inputs.flip(0) reverses the batch order.
        temp_x[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = inputs.flip(0)[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]
        # Update inputs with the CutMixed tensor.
        inputs = temp_x

        # Reshape to flattened (B, C_out, R^3) for LayerNorm and Transformer.
        inputs = torch.reshape(inputs, (-1, self.out_channels, self.resolution ** 3))
        # Permute to (B, R^3, C_out) for LayerNorm and Transformer.
        x = inputs.permute(0, 2, 1)

        # Apply Layer Normalization.
        x = self.layer_norm(x)
        # Add positional embedding.
        x += self.pos_embedding  # todo
        # Apply positional embedding dropout.
        x = self.pos_drop(x)
        # Pass through the Transformer.
        x = self.voxel_Trasformer(x) # Shape: (B, R^3, C_out)

        # Reshape back to 3D voxel grid format.
        x = torch.reshape(x, (-1, self.resolution, self.resolution, self.resolution, self.out_channels))

        # --- Apply CutMix to the output as well (or undo the input CutMix) ---
        # This step seems to apply the same CutMix operation to the output of the encoder.
        # This is a bit unusual for standard CutMix, which typically modifies inputs and labels.
        # It might be intended to ensure consistency if the model is expected to handle
        # CutMixed inputs throughout its layers, or to apply a similar regularization
        # at the output level.
        temp_x = x.clone()
        temp_x[:, bbx1:bbx2, bby1:bby2, bbz1:bbz2, :] = x.flip(0)[:, bbx1:bbx2, bby1:bby2, bbz1:bbz2, :]
        x = temp_x

        # Permute dimensions to (B, C_out, R, R, R) for consistency.
        x = x.permute(0, 4, 1, 2, 3)
        return x


class PVTConv(nn.Module):
    """
    PVTConv (Point-Voxel Transformer Convolution) is a hybrid architecture
    that processes 3D point cloud data by combining two parallel pathways:
    1. A **Voxel Path**: Converts points to a voxel grid, processes it with
       3D convolutions and windowed self-attention, and then devoxelizes back to points.
    2. A **Point Path**: Processes raw point features using a SharedTransformer
       (likely a point-based self-attention or MLP).

    The outputs of these two paths are then fused. This design aims to leverage
    the strengths of both grid-based (local context, convolution efficiency)
    and point-based (fine-grained detail, permutation invariance) processing.
    """
    def __init__(self, in_channels, out_channels, kernel_size, resolution, normalize=True, eps=0):
        """
        Initializes the PVTConv module.

        Args:
            in_channels (int): Number of input feature channels for points.
            out_channels (int): Number of output feature channels for points.
            kernel_size (int): Kernel size for initial voxel convolution.
            resolution (int): Resolution of the 3D voxel grid.
            normalize (bool): Whether to normalize coordinates in Voxelization.
            eps (float): Epsilon for normalization in Voxelization.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.boxsize = 3 # Default box size for windowed attention
        self.mlp_dims = out_channels # Default MLP hidden dimension
        self.drop_path1 = 0.1 # DropPath rate for Transformer
        self.drop_path2 = 0.2 # DropPath rate for Transformer

        # Voxelization module: Converts point features and coordinates into a dense voxel grid.
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)

        # Voxel Encoder: Processes the voxel grid using 3D convolutions and Transformer blocks.
        self.voxel_encoder = VoxelEncoder(in_channels, out_channels, kernel_size, resolution, self.boxsize,
                                          self.mlp_dims, self.drop_path1, self.drop_path2)

        # SE3d (Squeeze-and-Excitation) module: Applies channel-wise recalibration
        # to the voxel features, enhancing important channels and suppressing less relevant ones.
        self.SE = SE3d(out_channels)

        # SharedTransformer for point-based features: Processes the original point features.
        # This module is likely a point-based self-attention or an MLP that operates
        # directly on the point cloud.
        self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        """
        Performs the forward pass of the PVTConv module.

        Args:
            inputs (tuple): A tuple containing:
                            - features (torch.Tensor): (B, C_in, N) point-wise features.
                            - coords (torch.Tensor): (B, 3, N) original point coordinates (float).

        Returns:
            tuple: A tuple containing:
                   - fused_features (torch.Tensor): (B, C_out, N) fused per-point features.
                   - coords (torch.Tensor): (B, 3, N) original point coordinates (unchanged).
        """
        # Unpack the input tuple into point features and coordinates.
        features, coords = inputs # features: (B, C_in, N), coords: (B, 3, N)

        # -------------------------------
        # 1) Voxel path: Convert points → voxels → process → devoxelize back to points
        # -------------------------------

        # a) Voxelization: Bin points into a dense R³ grid.
        #    This operation aggregates point features into voxel cells (e.g., by averaging).
        #    - voxel_features: (B, C_voxel, R, R, R) - the 3D voxel grid features.
        #    - voxel_coords:   (B, N, 3) - integer voxel indices for each original point.
        #                      These are crucial for the devoxelization step.
        voxel_features, voxel_coords = self.voxelization(features, coords)

        # b) Voxel encoder: Apply 3D convolutions and local-window self-attention.
        #    This module processes the dense voxel grid, extracts hierarchical features,
        #    and aggregates local context within the grid.
        #    The output remains a 3D voxel grid of shape (B, C_voxel, R, R, R).
        voxel_features = self.voxel_encoder(voxel_features)

        # c) Squeeze-and-Excitation (SE): Channel-wise gating.
        #    This module adaptively re-weights each channel of the voxel features,
        #    selectively emphasizing informative channels.
        voxel_features = self.SE(voxel_features)

        # d) Trilinear devoxelization: Interpolate voxel grid features back to the original N points.
        #    This uses the `voxel_coords` (integer indices of the voxels each point belongs to)
        #    to perform trilinear interpolation, effectively sampling features from the continuous
        #    voxel grid at the exact locations of the original points.
        #    The output is per-point features of shape (B, C_voxel, N).
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)

        # -------------------------------
        # 2) Point path: Pure point-cloud self-attention
        # -------------------------------

        # a) Rearrange coordinates to (B, N, 3) for pairwise difference computation.
        #    The original 'coords' are (B, 3, N).
        pos = coords.permute(0, 2, 1) # Shape: (B, N, 3)

        # b) Compute pairwise relative positions.
        #    This calculates the vector difference between all pairs of points within a batch.
        #    - pos[:, :, None, :] is (B, N, 1, 3)
        #    - pos[:, None, :, :] is (B, 1, N, 3)
        #    Subtraction results in (B, N, N, 3), where (i, j) element is (pos[i] - pos[j]).
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        # Sum over the last dimension (3, for x,y,z components) to get a scalar bias for each pair.
        # This yields a (B, N, N) matrix, where each element (i, j) is a scalar bias
        # representing the "distance" or "relation" between point i and point j.
        rel_pos = rel_pos.sum(dim=-1) # Shape: (B, N, N)

        # Apply the SharedTransformer to the original point features, incorporating relative positions.
        # This path directly processes the point cloud, often using self-attention
        # that can leverage the relative positional information.
        # The output 'point_out' has shape (B, C_out, N).
        point_out = self.point_features(features, rel_pos)

        # -------------------------------
        # 3) Fuse voxel + point outputs
        # -------------------------------

        # Element-wise sum of the features from the two parallel paths.
        # This combines the spatially contextualized features from the voxel path
        # with the fine-grained, permutation-invariant features from the point path.
        # Both 'voxel_features' and 'point_out' are (B, C_out, N).
        fused_features = voxel_features + point_out

        # Return the fused per-point features and the original (unchanged) coordinates.
        return fused_features, coords

class PartPVTConv(nn.Module):
    """
    A variant of PVTConv, possibly tailored for part segmentation or other tasks
    where the point path might be simpler (e.g., not using relative positions).
    The main difference from PVTConv is in the `point_features` module's forward call.
    """
    def __init__(self, in_channels, out_channels, kernel_size, resolution, normalize=True, eps=0):
        """
        Initializes PartPVTConv. Parameters are similar to PVTConv.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.boxsize = 3
        self.mlp_dims = out_channels
        self.drop_path1 = 0.1
        self.drop_path2 = 0.1 # Slightly different drop_path2 from PVTConv

        # Voxelization, VoxelEncoder, SE3d are the same as in PVTConv.
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        self.voxel_encoder = VoxelEncoder(in_channels, out_channels, kernel_size, resolution, self.boxsize,
                                             self.mlp_dims, self.drop_path1, self.drop_path2)
        self.SE = SE3d(out_channels)
        # SharedTransformer for point-based features.
        self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        """
        Performs the forward pass of PartPVTConv.

        Args:
            inputs (tuple): A tuple containing:
                            - features (torch.Tensor): (B, C_in, N) point-wise features.
                            - coords (torch.Tensor): (B, 3, N) original point coordinates.

        Returns:
            tuple: A tuple containing:
                   - fused_features (torch.Tensor): (B, C_out, N) fused per-point features.
                   - coords (torch.Tensor): (B, 3, N) original point coordinates (unchanged).
        """
        features, coords = inputs
        # Voxel path is identical to PVTConv.
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_encoder(voxel_features)
        voxel_features = self.SE(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)

        # Fusion: The key difference is here.
        # The `SharedTransformer` in this variant is called with only `features`,
        # implying it might not use relative positional information or has a
        # simpler attention mechanism compared to PVTConv's point path.
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords

class SemPVTConv(nn.Module):
    """
    A variant of PVTConv, likely designed for semantic segmentation tasks.
    It incorporates the `SegVoxelEncoder`, which includes a CutMix-like
    data augmentation strategy.
    """
    def __init__(self, in_channels, out_channels, kernel_size, resolution, normalize=True, eps=0):
        """
        Initializes SemPVTConv. Parameters are similar to PVTConv.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.boxsize = 4 # Different box size (4)
        self.mlp_dims = out_channels * 4 # Larger MLP hidden dimension
        self.drop_path1 = 0. # Different DropPath rates
        self.drop_path2 = 0.1

        # Voxelization module.
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)

        # Voxel Encoder: Uses SegVoxelEncoder, which includes CutMix augmentation.
        self.voxel_encoder = SegVoxelEncoder(in_channels, out_channels, kernel_size, resolution, self.boxsize,
                                             self.mlp_dims, self.drop_path1, self.drop_path2)

        # SE3d module.
        self.SE = SE3d(out_channels)
        # SharedTransformer for point-based features (same as PartPVTConv, no rel_pos).
        self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        """
        Performs the forward pass of SemPVTConv.

        Args:
            inputs (tuple): A tuple containing:
                            - features (torch.Tensor): (B, C_in, N) point-wise features.
                            - coords (torch.Tensor): (B, 3, N) original point coordinates.

        Returns:
            tuple: A tuple containing:
                   - fused_features (torch.Tensor): (B, C_out, N) fused per-point features.
                   - coords (torch.Tensor): (B, 3, N) original point coordinates (unchanged).
        """
        features, coords = inputs
        # Voxel path: Uses SegVoxelEncoder which applies CutMix.
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_encoder(voxel_features) # CutMix happens here
        voxel_features = self.SE(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)

        # Point path and fusion: Identical to PartPVTConv.
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords
