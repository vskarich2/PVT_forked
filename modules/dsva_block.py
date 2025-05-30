import torch
import torch.nn as nn
import numpy as np
from timm.layers import DropPath

from PVT_forked_repo.PVT_forked.modules.box_attention import BoxAttention
from PVT_forked_repo.PVT_forked.modules.dsva_cross_attention import SparseDynamicVoxelAttention
from PVT_forked_repo.PVT_forked.modules.feed_forward import FeedForward

import torch
from torch_cluster import knn_graph

def compute_knn_graph(voxel_centers, k):
    """
    Compute batched kNN graph from voxel centers.

    Args:
        voxel_centers (Tensor): shape (B, V, 3) — normalized voxel centers
        k (int): number of neighbors

    Returns:
        edge_index_list: list of (2, B*V*k) tensors, one per batch
    """
    B, V, _ = voxel_centers.shape
    edge_index_list = []

    for b in range(B):
        x = voxel_centers[b]  # shape (V, 3)
        edge_index = knn_graph(x, k=k, loop=False)  # shape: (2, V*k)
        edge_index_list.append(edge_index)

    return edge_index_list

def generate_voxel_grid_centers(resolution, device='cpu'):
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

    return grid.to(device)  # shape (V, 3)

def generate_top_k_neighbors():
    voxel_centers = generate_voxel_grid_centers(resolution=30).unsqueeze(0).expand(8, -1, -1)  # (8, 27000, 3)

    # edges[0] shape → (2, 270000), where 270000 = 27000 × 10
    edges = compute_knn_graph(voxel_centers, k=100)


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

def extract_non_empty_voxel_mask(voxel_grid):
    B, C, R, _, _ = voxel_grid.shape
    V = R ** 3
    flat_voxel_grid = voxel_grid.view(B, C, V)
    non_empty_mask = flat_voxel_grid.abs().sum(dim=1) > 0
    return non_empty_mask

def filter_voxels_by_mask(tensor, mask):
    return [t[m] for t, m in zip(tensor, mask)]

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
        self.heads = 4 # Fixed number of attention heads for this block
        self.dim_head = self.out_channels // self.heads # Dimension per head

        B = self.args.batch_size

        # Initialize the DSVA Cross Attention module.
        self.attn  = SparseDynamicVoxelAttention(dim=64, num_heads=4, k_knn=10, k_select=4)

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
        Performs the forward pass of the DSVABlock.

        Args:
            inputs (torch.Tensor): Input tensor of shape (B, R^3, C_out),
                                   where R^3 is the flattened resolution.

        Returns:
            torch.Tensor: Output tensor of the same shape (B, R^3, C_out).
        """
        shortcut = inputs # Store input for residual connection
        batch_size = inputs.shape[0] # Batch size

        # Apply LayerNorm to the input.
        x = self.norm1(inputs)
        # Reshape from flattened (B, R^3, C) to 3D (B, R, R, R, C) for spatial operations.
        x = torch.reshape(x, (batch_size, self.resolution, self.resolution, self.resolution, self.out_channels))

        # Apply DSVA Attention
        self.attn(voxel_tokens, voxel_coords, non_empty_mask)

        # Output shape: (-1, box_size^3, C_out)

        # First residual connection and DropPath.
        # Applies DropPath to the attention output, scales by 0.5 (common in some architectures),
        # and adds it to the original shortcut (input).
        x = self.drop_path(x) * 0.5 + shortcut

        # Second residual connection and DropPath, after MLP.
        # Apply LayerNorm, then MLP, then DropPath, scale by 0.5, and add to 'x'.
        x = self.drop_path(self.mlp(self.norm2(x))) * 0.5 + x

        return x