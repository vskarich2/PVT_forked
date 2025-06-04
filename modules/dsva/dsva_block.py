import torch.nn as nn
from timm.layers import DropPath

from modules.dsva.dsva_cross_attention import SparseDynamicVoxelAttention
from modules.feed_forward import FeedForward

import torch


def map_sparse_to_dense(updated_tokens, non_empty_mask, V, D):
    """
    Maps a list of sparse token outputs back to a dense grid.

    Args:
        updated_tokens: list of (V', D) tensors — one per batch
        non_empty_mask: (B, V) boolean mask of non-empty voxels
        V: int — total number of voxels per sample (e.g., R³)
        D: int — embedding dimension

    Returns:
        dense_tokens: (B, V, D) tensor with zeros in empty voxels
    """
    B = len(updated_tokens)
    dense_tokens = torch.zeros(B, V, D, device=updated_tokens[0].device)

    for b in range(B):
        dense_tokens[b, non_empty_mask[b]] = updated_tokens[b]

    return dense_tokens

def generate_voxel_grid_centers(resolution, args):
    """
    Returns a tensor of shape (V, 3) with the 3D center coordinates
    of each voxel in a cubic grid of shape (R x R x R), normalized to [-1, 1]^3.

    Args:
        resolution (int): Number of voxels per axis (R)
        device (str or torch.device): Where to place the resulting tensor

    Returns:
        Tensor: (V, 3), where V = R^3
    """
    # Generate voxel indices in 3D grid: shape (R, R, R, 3)
    grid = torch.stack(torch.meshgrid(
        torch.arange(resolution),
        torch.arange(resolution),
        torch.arange(resolution),
        indexing='ij'  # Makes indexing (x, y, z) order
    ), dim=-1).float()  # shape: (R, R, R, 3)

    # Reshape to flat list of voxel indices: (V, 3) where V = R^3
    grid = grid.reshape(-1, 3)  # e.g., (27000, 3) for R=30

    # Compute voxel centers in normalized [-1, 1]^3 space
    grid = (grid + 0.5) / resolution  # Normalize to (0, 1)
    grid = grid * 2 - 1  # Map to (-1, 1)

    voxel_centers = grid.unsqueeze(0).expand(args.batch_size, -1, -1)  # shape: (B, R^3, 3)
    return voxel_centers.to(args.device)

def reconstruct_dense_masked_scatter_2(updated_list, mask, original):
    """
    updated_list: list of length B, each entry is (V_b, C) in whatever dtype
    mask:        BoolTensor of shape (B, V)
    original:    Tensor of shape (B, V, C) (used only for shape/device)

    We want `out` to live in the same dtype as updated_list[0], not necessarily original.dtype.
    """
    B, V, C = original.shape

    # Determine the dtype from updated_list[0] (if the list is nonempty). Fallback to original.dtype.
    if len(updated_list) > 0 and updated_list[0].numel() > 0:
        target_dtype = updated_list[0].dtype
    else:
        target_dtype = original.dtype

    # Create the output tensor using that target_dtype
    out = torch.zeros(B, V, C, device=original.device, dtype=target_dtype)

    for b in range(B):
        # updated_list[b] already has dtype=target_dtype (half under autocast),
        # so this assignment will no longer complain.
        out[b, mask[b], :] = updated_list[b]

    return out

def reconstruct_dense_masked_scatter(updated_list, mask, original):
    # 1) Clone the original so we don’t overwrite it in-place (optional)
    out = original.clone()

    B, V, C = original.shape
    for b in range(B):
        # Note that mask[b].sum() == updated_list[b].size(0) == Vb
        out[b, mask[b], :] = updated_list[b]

    return out

class DSVABlock(nn.Module):
    """
    A single DSVA block
    """
    def __init__(
            self,
            args,
            out_channels,
            resolution,
            mlp_dims,
            drop_path=0.):
        """
        Initializes the DSVABlock.

        Args:
            out_channels (int): The feature dimension (C) of the input/output.
            resolution (int): The resolution (R) of the 3D voxel grid.
            boxsize (int): The size of the cubic attention window.
            mlp_dims (int): The hidden dimension for the FeedForward network.
            shift (bool): If True, enables shifted window attention. If None, no shifting.
            drop_path (float): Dropout rate for DropPath (stochastic depth).
        """
        super().__init__()
        self.args = args
        self.out_channels = out_channels
        self.resolution = resolution

        if args.eight_heads:
            self.heads = 4 if resolution == 30 else 8
        else:
            self.heads = 4

        self.dim_head = self.out_channels // self.heads # Dimension per head

        knn_size = args.knn_size_fine if resolution == 30 else args.knn_size_coarse
        top_k_select = args.top_k_select_fine if resolution == 30 else args.top_k_select_coarse

        # Initialize the DSVA Cross Attention module.
        self.attn  = SparseDynamicVoxelAttention(
            dim=self.out_channels,
            num_heads=self.heads,
            knn_size=knn_size,
            top_k_select=top_k_select,
        )

        # Layer Normalization layers. Applied before attention and MLP.
        self.norm1 = nn.LayerNorm(out_channels) # For input to attention
        self.norm2 = nn.LayerNorm(out_channels) # For input to MLP

        # FeedForward Network (MLP).
        self.mlp_dim = mlp_dims
        self.mlp = FeedForward(out_channels, self.mlp_dim)

        # DropPath (Stochastic Depth): A regularization technique where paths
        # in the network are randomly dropped during training.
        self.drop_path = DropPath(drop_path)

    def forward(self, voxel_tokens, non_empty_mask):
        """
        Performs the forward pass of the DSVABlock.

        Args:
            voxel_tokens: (B, R^3, D) - voxel token embeddings
        Returns:
            torch.Tensor: Output tensor of the same shape (B, R^3, C_out).
        """
        shortcut = voxel_tokens # Store input for residual connection

        # Apply LayerNorm to the voxel_tokens.
        voxel_tokens = self.norm1(voxel_tokens)

        # voxel_coords: (B, R^3, 3) - voxel centers
        voxel_coords = generate_voxel_grid_centers(self.resolution, self.args)

        # Apply DSVA Attention
        enriched_tokens = self.attn(voxel_tokens, voxel_coords, non_empty_mask)

        # Need to reshape list of enriched token tensors
        # to flattened (B, R^3, C) for subsequent operations.
        x = reconstruct_dense_masked_scatter_2(enriched_tokens, non_empty_mask, voxel_tokens)

        # First residual connection and DropPath.
        # Applies DropPath to the attention output, scales by 0.5 (common in some architectures),
        # and adds it to the original shortcut (input).
        x = self.drop_path(x) * 0.5 + shortcut

        # Second residual connection and DropPath, after MLP.
        # Apply LayerNorm, then MLP, then DropPath, scale by 0.5, and add to 'x'.
        x = self.drop_path(self.mlp(self.norm2(x))) * 0.5 + x

        return x

class DSVABlockLarge(nn.Module):
    """
    A higher‐capacity DSVA block:
      - expands D → 2D for attention
      - uses a larger MLP hidden size (4D)
      - then projects back 2D → D
    """
    def __init__(
            self,
            args,
            out_channels,
            resolution,
            drop_path=0.
    ):
        """
        Args:
            args: must contain knn_size_fine, top_k_select_fine, knn_size_coarse, top_k_select_coarse,
                  batch_size, device
            out_channels (int): the original token dimension D
            resolution (int): grid size per axis (R)
            drop_path (float): drop probability for DropPath
        """
        super().__init__()
        self.args = args
        self.resolution = resolution
        self.D = out_channels                    # original feature dim
        self.embed_dim = out_channels * 2        # expanded dim for attention
        # keep same head‐count (you could increase heads if desired)
        self.heads = 4 if resolution == 30 else 8
        self.dim_head = self.embed_dim // self.heads

        # choose knn_size / top_k based on resolution
        knn_size = args.knn_size_fine if resolution == 30 else args.knn_size_coarse
        top_k_select = args.top_k_select_fine if resolution == 30 else args.top_k_select_coarse

        # 1) Project D → 2D before attention
        self.expand = nn.Linear(self.D, self.embed_dim)

        # 2) Sparse dynamic attention in the higher dim
        self.attn = SparseDynamicVoxelAttention(
            dim=self.embed_dim,
            num_heads=self.heads,
            knn_size=knn_size,
            top_k_select=top_k_select
        )

        # 3) Project 2D → D so the residual can add to the original tokens
        self.contract = nn.Linear(self.embed_dim, self.D)

        # LayerNorm operates in the expanded space
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim)

        # FeedForward MLP in the expanded space, with hidden size 4D
        self.mlp = FeedForward(self.embed_dim, self.embed_dim * 2)

        self.drop_path = DropPath(drop_path)

    def forward(self, voxel_tokens, non_empty_mask):
        """
        Args:
            voxel_tokens: (B, V, D)
            non_empty_mask: (B, V) bool mask of which voxels exist
        Returns:
            (B, V, D) after one DSVA block with higher capacity
        """
        B, V, D = voxel_tokens.shape
        assert D == self.D, f"Expected D={self.D}, got {D}"
        shortcut = voxel_tokens  # (B, V, D)

        # --- 1) Expand & LayerNorm ---
        x = self.expand(voxel_tokens)     # → (B, V, 2D)
        x = self.norm1(x)                 # LN over last dimension (2D)

        # --- 2) Compute voxel centers and apply sparse attention ---
        voxel_coords = generate_voxel_grid_centers(self.resolution, self.args)  # (B, V, 3)
        enriched = self.attn(x, voxel_coords, non_empty_mask)
        # enriched: list length B; enriched[b].shape = (Vb, 2D)

        # 3) Scatter enriched back into dense (B, V, 2D)
        x = reconstruct_dense_masked_scatter_2(enriched, non_empty_mask, x)  # (B, V, 2D)

        # --- 4) Project back to D for the first residual ---
        x = self.contract(x)   # → (B, V, D)
        x = self.drop_path(x) * 0.5 + shortcut  # residual #1: combine with original D

        # --- 5) Second sub-layer: MLP in expanded space ---
        x2 = self.expand(x)    # → (B, V, 2D)   (re-expand)
        x2 = self.norm2(x2)    # LN in 2D space
        x2 = self.mlp(x2)      # FeedForward in 2D space: (B, V, 2D)

        # 6) Contract back to D for residual #2
        x2 = self.contract(x2) # → (B, V, D)
        x = self.drop_path(x2) * 0.5 + x   # residual #2

        return x