"""
modules/dynamic_sparse.py

Utilities for dynamic sparse kNN graph construction and edge-feature assembly
in voxel-based PVTConv.

Functions:
- knn_filter_empty: kNN among non-empty voxels only (handles cases where #non-empty voxels < k).
- build_edge_features: Concatenate anchor feats, neighbor feats, and normalized offsets.
- EdgeScorer: Simple 2-layer MLP to score edge features.
"""

import torch
import torch.nn as nn


def knn_filter_empty(voxel_coords: torch.Tensor,
                     voxel_feats: torch.Tensor,
                     k: int,
                     eps: float = 1e-6):
    """
    Compute k-nearest neighbors only among non-empty voxels (feature norm > eps).
    Gracefully handles cases where the number of non-empty voxels N_b is less than k.

    Args:
        voxel_coords: Tensor of shape (B, V, 3) with 3D coordinates of each voxel center.
        voxel_feats:  Tensor of shape (B, V, D) with learned token embeddings per voxel.
        k:            Number of neighbors to retrieve per voxel.
        eps:          Threshold under which a voxel is considered empty.

    Returns:
        neighbors: LongTensor of shape (B, V, k) with neighbor indices,
                   or -1 for empty voxels or padded slots when N_b < k.

    Notes:
    - Total potential edges per batch = V * k (e.g. 30³ * 100 = 2.7M edges).
    - Empty voxels (feature-norm <= eps) are skipped in the distance search.
    """
    # Get batch size and total voxel count
    B, V, _ = voxel_coords.shape
    # Print summary: voxel count and max edges
    print(f"[dynamic_sparse] B={B}, V={V}, k={k}  # max edges per batch = V*k = {V*k}")

    device = voxel_coords.device
    # Create mask for non-empty voxels
    mask = voxel_feats.norm(dim=-1) > eps  # shape (B, V)
    # Initialize neighbors with -1 (invalid placeholder)
    neighbors = torch.full((B, V, k), -1, dtype=torch.long, device=device)

    for b in range(B):
        # Indices of non-empty voxels in this batch
        nonzero_idx = mask[b].nonzero(as_tuple=False).squeeze(1)
        N_b = nonzero_idx.numel()
        if N_b == 0:
            # No non-empty voxels: skip
            continue
        # Actual number of neighbors to compute for this batch
        k_b = min(k, N_b)

        # Extract coordinates of non-empty voxels
        coords_b = voxel_coords[b, nonzero_idx]  # (N_b, 3)
        # Compute pairwise distances among them
        dists = torch.cdist(coords_b.unsqueeze(0), coords_b.unsqueeze(0)).squeeze(0)  # (N_b, N_b)
        # Exclude self-distances
        diag = torch.arange(N_b, device=device)
        dists[diag, diag] = float('inf')
        # Get k_b nearest neighbor indices per non-empty voxel
        topk = torch.topk(dists, k_b, dim=-1, largest=False).indices  # (N_b, k_b)

        # Map local indices back to global voxel indices
        global_neighbors = nonzero_idx[topk]  # (N_b, k_b)
        # Fill into the neighbors tensor; rows for empty voxels remain -1
        neighbors[b, nonzero_idx, :k_b] = global_neighbors

    return neighbors


def build_edge_features(voxel_feats: torch.Tensor,
                        voxel_coords: torch.Tensor,
                        neighbors: torch.Tensor):
    """
    Assemble per-edge feature vectors [h_i || h_j || normalized_offset] for each kNN edge.

    Args:
        voxel_feats:  Tensor of shape (B, V, D) with voxel token embeddings.
        voxel_coords: Tensor of shape (B, V, 3) with 3D centers of each voxel.
        neighbors:    LongTensor of shape (B, V, k) with neighbor indices.

    Returns:
        edge_feats: Tensor of shape (B, V, k, 2*D+3) of concatenated features.

    Notes:
    - Number of edge feature vectors = B * V * k (max edges per batch).
    - Each feature = [h_i (D) || h_j (D) || offset_norm (3)] => 2D+3 dims.
    """
    B, V, D = voxel_feats.shape
    _, _, k = neighbors.shape

    # Anchor features: repeat each h_i across k neighbors -> (B, V, k, D)
    h_i = voxel_feats.unsqueeze(2).expand(-1, -1, k, -1)

    # Gather neighbor features; clamp negative indices to 0 (they map to dummy)
    idx_feats = neighbors.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, D)
    feats_exp = voxel_feats.unsqueeze(1).expand(-1, V, -1, -1)
    h_j = torch.gather(feats_exp, dim=2, index=idx_feats)  # (B, V, k, D)

    # Gather coordinates for computing offsets
    c_i = voxel_coords.unsqueeze(2).expand(-1, -1, k, -1)    # (B, V, k, 3)
    idx_coords = neighbors.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, 3)
    coords_exp = voxel_coords.unsqueeze(1).expand(-1, V, -1, -1)
    c_j = torch.gather(coords_exp, dim=2, index=idx_coords)  # (B, V, k, 3)

    # Compute normalized offsets (zeroed for invalid neighbor indices)
    offsets = c_j - c_i
    norm = offsets.norm(dim=-1, keepdim=True)
    offset_norm = offsets / (norm + 1e-6)
    # Mask out offsets for padded slots (neighbors == -1)
    offset_norm = torch.where(neighbors.unsqueeze(-1) < 0,
                              torch.zeros_like(offset_norm),
                              offset_norm)

    # Concatenate [h_i || h_j || offset_norm] => (B, V, k, 2*D+3)
    edge_feats = torch.cat([h_i, h_j, offset_norm], dim=-1)
    return edge_feats


class EdgeScorer(nn.Module):
    """
    Simple 2-layer MLP to produce a scalar score for each edge feature.

    Input shape: (B, V, k, F) where F = 2*D+3.
    Output shape: (B, V, k) containing one score per edge.
    """
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # maps 2D+3 -> hidden
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)        # hidden -> scalar score
        )

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        # Reshape to (B*V*k, F) for MLP, then back to (B, V, k)
        B, V, k, F = edge_feats.shape
        x = edge_feats.reshape(B * V * k, F)
        scores = self.net(x).view(B, V, k)
        return scores
