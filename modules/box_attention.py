import torch
import torch.nn as nn
import numpy as np

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

