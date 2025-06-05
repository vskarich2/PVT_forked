"""
modules/dynamic_sparse.py

Utilities for dynamic sparse kNN graph construction and edge-feature assembly
in voxel-based PVTConv. This module focuses on efficiency by only considering
voxels that contain data ("non-empty" voxels) when building graph connections.

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
    Gracefully handles cases where the number of non-empty voxels N_b (per batch item) is less than k.

    Args:
        voxel_coords: Tensor of shape (B, V, 3) with 3D coordinates of each voxel center.
                      These coordinates are used to calculate distances for kNN.
        voxel_feats:  Tensor of shape (B, V, D) with learned token embeddings per voxel.
                      Used to determine if a voxel is "empty" based on its feature norm.
        k:            Number of neighbors to retrieve per voxel. This is the target 'k' for kNN.
        eps:          Threshold under which a voxel is considered empty.
                      If voxel_feats.norm() <= eps, the voxel is ignored in kNN.

    Returns:
        neighbors: LongTensor of shape (B, V, k) with global neighbor indices.
                   Contains -1 for:
                   a) Voxels that were initially empty.
                   b) Valid voxels for which fewer than k neighbors were found (padded slots).

    Notes:
    - Total potential edges per batch = V * k (e.g. 30³ * 100 = 2.7M edges).
      This highlights the need for sparse computation; processing all potential edges
      would be very expensive if most voxels are empty.
    - Empty voxels (feature-norm <= eps) are skipped in the distance search.
      WHY: This is the core of "dynamic sparse" kNN. It significantly reduces
      computational cost and memory usage by only considering voxels that carry
      meaningful information (features). This is especially effective for sparse
      3D data where many voxels in a grid might not intersect with the actual object.
    """
    # Get batch size (B) and total number of voxels (V) per item in the batch.
    B, V, _ = voxel_coords.shape
    # Print summary: voxel count and max edges. Useful for debugging and understanding scale.
    # print(f"[dynamic_sparse] B={B}, V={V}, k={k}  # max edges per batch = V*k = {V*k}") # Usually commented out in production

    device = voxel_coords.device  # Ensure all tensors are on the same device.

    # --- Identify non-empty voxels ---
    # Purpose: To create a boolean mask indicating which voxels are non-empty.
    # A voxel is non-empty if the L2 norm of its feature vector is greater than eps.
    # WHY `eps`: Provides a small tolerance to avoid issues with floating-point precision
    # and allows defining a threshold for what constitutes a "meaningful" feature vector.
    mask = voxel_feats.norm(dim=-1) > eps  # shape (B, V)

    # --- Initialize neighbors tensor ---
    # Purpose: To create a tensor to store the indices of the k nearest neighbors for each voxel.
    # It's initialized with -1, which serves as a placeholder for:
    #   1. Neighbors of initially empty voxels (their rows will remain all -1s).
    #   2. Padded neighbor slots if a non-empty voxel has fewer than k other non-empty voxels.
    # WHY -1: It's an invalid index, making it easy to identify and handle these cases later
    # (e.g., in `build_edge_features` or attention mechanisms).
    neighbors = torch.full((B, V, k), -1, dtype=torch.long, device=device)

    # --- Process each item in the batch independently ---
    # WHY loop over batch: kNN is typically performed independently for each point cloud/voxel grid
    # in the batch. The set of non-empty voxels and their spatial relationships can vary
    # significantly between batch items.
    for b in range(B):
        # --- Select non-empty voxels for the current batch item ---
        # Purpose: To get the 1D indices of non-empty voxels within the current batch item.
        # `mask[b]` is a boolean tensor of shape (V,).
        # `.nonzero(as_tuple=False)` returns a 2D tensor where each row is an index of a True value.
        # `.squeeze(1)` converts it to a 1D tensor of indices.
        nonzero_idx = mask[b].nonzero(as_tuple=False).squeeze(1)  # Shape: (N_b,) where N_b is num non-empty
        N_b = nonzero_idx.numel()  # Number of non-empty voxels in this batch item.

        # Purpose: If there are no non-empty voxels, there's nothing to compute kNN for.
        if N_b == 0:
            continue  # Skip to the next batch item.

        # --- Adjust k for the current batch item ---
        # Purpose: To determine the actual number of neighbors to find (`k_b`).
        # If the number of non-empty voxels (`N_b`) is less than the desired `k`,
        # we can only find `N_b - 1` neighbors (excluding self) or `N_b` if self-loops are allowed but typically not for kNN.
        # Here, it implies we can find at most `N_b` neighbors if `k` is large, or `k` if `N_b` is sufficient.
        # The kNN below will find `min(k, N_b-1)` if self-loops are strictly excluded by distance.
        # This line ensures `topk` doesn't request more neighbors than available distinct points.
        # If N_b=1, k_b will be 1, topk will try to find 1 neighbor.
        k_actual_for_topk = min(k, N_b - 1)  # Max neighbors to find, excluding self.
        if N_b <= 1:  # If only one or zero non-empty voxels, no distinct neighbors can be found.
            k_actual_for_topk = 0

        # Extract coordinates of only the non-empty voxels for this batch item.
        # WHY: This significantly reduces the size of the distance matrix calculation.
        coords_b = voxel_coords[b, nonzero_idx]  # Shape: (N_b, 3)

        # If there are not enough non-empty voxels to find any neighbors, skip kNN computation.
        if k_actual_for_topk == 0:
            # The `neighbors` tensor for these `nonzero_idx` voxels will remain -1, which is correct.
            continue

        # --- Compute pairwise distances among non-empty voxels ---
        # Purpose: To calculate the Euclidean distance between all pairs of non-empty voxels.
        # `coords_b.unsqueeze(0)` makes it (1, N_b, 3) for cdist compatibility if needed,
        # but cdist can handle (N_b, 3) vs (N_b, 3) directly.
        # `torch.cdist` is an efficient way to compute pairwise distances.
        dists = torch.cdist(coords_b, coords_b)  # Shape: (N_b, N_b)

        # --- Exclude self-distances ---
        # Purpose: To prevent a voxel from being its own nearest neighbor.
        # This is done by setting the diagonal elements of the distance matrix to infinity.
        diag_indices = torch.arange(N_b, device=device)
        dists[diag_indices, diag_indices] = float('inf')

        # --- Find k_b nearest neighbors ---
        # Purpose: For each non-empty voxel, find the indices of its `k_actual_for_topk` closest neighbors
        # based on the computed distances.
        # `largest=False` ensures we get the smallest distances (nearest neighbors).
        # `.indices` gives the indices of these neighbors within the `coords_b` tensor (i.e., local indices).
        topk_indices = torch.topk(dists, k_actual_for_topk, dim=-1,
                                  largest=False).indices  # Shape: (N_b, k_actual_for_topk)

        # --- Map local neighbor indices back to global voxel indices ---
        # Purpose: The `topk_indices` are local to the `N_b` non-empty voxels.
        # We need to map them back to the original V-dimensional voxel indexing.
        # `nonzero_idx` holds the global indices of the non-empty voxels.
        # So, `nonzero_idx[topk_indices]` performs this mapping.
        global_neighbor_indices = nonzero_idx[topk_indices]  # Shape: (N_b, k_actual_for_topk)

        # --- Store the global neighbor indices ---
        # Purpose: To fill the `neighbors` tensor with the computed global neighbor indices.
        # `neighbors[b, nonzero_idx, :k_actual_for_topk]` selects the rows corresponding to the
        # non-empty voxels for the current batch item and fills the first `k_actual_for_topk` neighbor slots.
        # If `k_actual_for_topk < k`, the remaining slots for these voxels will keep their -1 padding.
        neighbors[b, nonzero_idx, :k_actual_for_topk] = global_neighbor_indices

    return neighbors


def build_edge_features(voxel_feats: torch.Tensor,
                        voxel_coords: torch.Tensor,
                        neighbors: torch.Tensor):
    """
    Assemble per-edge feature vectors [h_i || h_j || normalized_offset] for each kNN edge.
    This is a common way to represent edges in graph neural networks, capturing information
    from both connected nodes and their relative spatial arrangement.

    Args:
        voxel_feats:  Tensor of shape (B, V, D) with voxel token embeddings (h).
        voxel_coords: Tensor of shape (B, V, 3) with 3D centers of each voxel (c).
        neighbors:    LongTensor of shape (B, V, k) with global neighbor indices.
                      Contains -1 for invalid/padded neighbor slots.

    Returns:
        edge_feats: Tensor of shape (B, V, k, 2*D+3) of concatenated features.
                    For invalid neighbors (index -1), the h_j and offset parts
                    will effectively be zeroed out or based on a dummy voxel at index 0.

    Notes:
    - Number of edge feature vectors = B * V * k (max edges per batch).
    - Each feature = [h_i (D) || h_j (D) || offset_norm (3)] => 2D+3 dims.
      - `h_i`: Features of the source/anchor voxel.
        WHY: Provides the context of the "sending" node of the edge.
      - `h_j`: Features of the target/neighbor voxel.
        WHY: Provides the context of the "receiving" node of the edge.
      - `offset_norm`: Normalized 3D spatial offset from anchor to neighbor (c_j - c_i).
        WHY: Encodes the relative geometric relationship. Normalization makes it
        scale-invariant and can lead to more stable learning.
    """
    B, V, D = voxel_feats.shape  # Batch size, Num voxels, Feature dimension
    _, _, k = neighbors.shape  # Num neighbors

    # --- Anchor features (h_i) ---
    # Purpose: To get the features of the source voxel (h_i) for each of its k potential edges.
    # `voxel_feats.unsqueeze(2)` adds a new dimension for k: (B, V, 1, D).
    # `.expand(-1, -1, k, -1)` repeats the features k times along this new dimension.
    # So, for each voxel `i`, its feature vector `h_i` is replicated `k` times.
    h_i = voxel_feats.unsqueeze(2).expand(-1, -1, k, -1)  # Shape: (B, V, k, D)

    # --- Gather neighbor features (h_j) ---
    # Purpose: To get the features of the k neighbor voxels (h_j) for each source voxel `i`.
    # This is done using `torch.gather`, which is an advanced indexing operation.

    # `neighbors.clamp(min=0)`:
    #   WHY: The `neighbors` tensor can contain -1 for invalid/padded slots.
    #   `torch.gather` requires non-negative indices. Clamping -1 to 0 means that
    #   invalid neighbors will effectively gather features from voxel 0. This "dummy"
    #   feature will then be processed, but its effect can be nullified later if needed
    #   (e.g., if voxel 0 is guaranteed to be empty or if offsets are zeroed).
    #   A more robust approach might involve masking out these edges entirely,
    #   but using index 0 is a common simplification if voxel 0's features are zero
    #   or if the subsequent offset masking handles it.
    idx_for_feats_gather = neighbors.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, D)  # Shape: (B, V, k, D)

    # `voxel_feats.unsqueeze(1).expand(-1, V, -1, -1)` is not strictly needed if `gather` is used carefully.
    # A more direct way to prepare for gather:
    # We want to select from `voxel_feats` (B,V,D) using indices `idx_for_feats_gather` (B,V,k,D),
    # where the critical part of `idx_for_feats_gather` is the neighbor index for the V dimension.
    # `torch.gather(input, dim, index)`
    # input: `voxel_feats` (B, V_all, D)
    # index: `neighbors_clamped_expanded_for_D` (B, V_src, k, D_idx_placeholder)
    # We need to gather along dim=1 (the V_all dimension of voxel_feats).
    # To do this, `voxel_feats` needs to be shaped such that each of the V_src voxels can pick from V_all.
    # This typically means `voxel_feats` is broadcasted or repeated.
    # Let's consider `voxel_feats` as the source tensor: (B, V_total, D)
    # `idx_for_feats_gather` has shape (B, V_anchor, k, D_feat_dim_expansion)
    # The actual indices used for gathering along dim 1 of `voxel_feats` are in `neighbors.clamp(min=0)`.
    # We need to align dimensions.
    # `voxel_feats` is (B, V_global, D).
    # `h_j_indices` = `neighbors.clamp(min=0)` is (B, V_anchor, k).
    # We want `h_j` of shape (B, V_anchor, k, D).
    # We can iterate or use advanced indexing. `torch.gather` is the way.
    # The `idx_for_feats_gather` is correctly shaped to tell `gather` which D-dimensional vector to pick.
    # The `dim=1` for gather is incorrect. We are gathering *from* the global list of V voxels.
    # `voxel_feats` is (B, V, D). We need to select V' (neighbor) features for each of V voxels, k times.
    # A common pattern:
    batch_idx = torch.arange(B, device=voxel_feats.device).view(B, 1, 1).expand(B, V, k)
    # `neighbors_clamped` is (B, V, k)
    # `h_j = voxel_feats[batch_idx, neighbors.clamp(min=0)]` would give (B, V, k, D)
    h_j = voxel_feats[batch_idx, neighbors.clamp(min=0)]  # Shape: (B, V, k, D)

    # --- Gather coordinates for computing offsets (c_i and c_j) ---
    # `c_i`: Coordinates of the source/anchor voxel, expanded for k neighbors.
    c_i = voxel_coords.unsqueeze(2).expand(-1, -1, k, -1)  # Shape: (B, V, k, 3)

    # `c_j`: Coordinates of the neighbor voxels.
    # Similar to gathering `h_j`, we gather coordinates `c_j` using neighbor indices.
    # `idx_for_coords_gather` is shaped (B, V, k, 3) to pick 3D coord vectors.
    # idx_for_coords_gather = neighbors.clamp(min=0).unsqueeze(-1).expand(-1, -1, -1, 3)
    # `coords_exp` is not the most direct way.
    # c_j = torch.gather(voxel_coords.unsqueeze(1).expand(-1,V,-1,-1), dim=2, index=idx_for_coords_gather)
    c_j = voxel_coords[batch_idx, neighbors.clamp(min=0)]  # Shape: (B, V, k, 3)

    # --- Compute normalized offsets (c_j - c_i) ---
    # Purpose: To calculate the spatial vector from anchor `i` to neighbor `j`.
    offsets = c_j - c_i  # Shape: (B, V, k, 3)

    # Purpose: To normalize these offset vectors to unit length.
    # WHY Normalization:
    #   - Scale Invariance: Makes the feature robust to the absolute distances between voxels.
    #   - Stability: Prevents very large or very small offset values from dominating learning.
    #   - Directional Information: Emphasizes the direction of the neighbor relative to the anchor.
    # `keepdim=True` ensures `norm` has shape (B, V, k, 1) for broadcasting during division.
    # `+ 1e-6` (epsilon) prevents division by zero if an offset vector has zero length
    # (e.g., if c_j == c_i, which can happen if a clamped index 0 points to the same voxel).
    norm = offsets.norm(dim=-1, keepdim=True)
    offset_norm = offsets / (norm + 1e-6)  # Shape: (B, V, k, 3)

    # --- Mask out offsets for padded/invalid neighbor slots ---
    # Purpose: If a neighbor slot was padded (original neighbor index was -1),
    # its gathered `h_j` and `c_j` might be from voxel 0 (due to clamping).
    # The resulting offset might be non-zero and misleading.
    # This step explicitly sets the `offset_norm` to zero for such invalid edges.
    # `neighbors.unsqueeze(-1) < 0` creates a boolean mask for invalid edges.
    # `torch.where` selects `zeros_like(offset_norm)` for these invalid edges.
    offset_norm = torch.where(neighbors.unsqueeze(-1) < 0,  # Condition: is neighbor index < 0?
                              torch.zeros_like(offset_norm),  # Value if True (invalid neighbor)
                              offset_norm)  # Value if False (valid neighbor)

    # Also, for invalid neighbors, h_j should ideally be zeroed out or masked.
    # The current h_j for invalid neighbors is h_0 (features of voxel 0).
    # If voxel 0 is guaranteed to be empty (zero features), then h_j is already effectively zero.
    # Otherwise, a similar `torch.where` could be applied to `h_j`.
    h_j = torch.where(neighbors.unsqueeze(-1) < 0,
                      torch.zeros_like(h_j),
                      h_j)

    # --- Concatenate features to form edge features ---
    # Purpose: To combine `h_i`, `h_j`, and `offset_norm` into a single feature vector for each edge.
    # `dim=-1` concatenates along the last dimension (the feature dimension).
    edge_feats = torch.cat([h_i, h_j, offset_norm], dim=-1)  # Shape: (B, V, k, D + D + 3)
    return edge_feats


class EdgeScorer(nn.Module):
    """
    Simple 2-layer MLP to produce a scalar score for each edge feature.
    This score can represent the importance, relevance, or strength of an edge.

    Input shape: (B, V, k, F_in) where F_in = 2*D+3 (dimension of edge_feats).
    Output shape: (B, V, k) containing one scalar score per edge.

    WHY use an MLP for scoring:
    - Learned Interaction: Allows the model to learn complex interactions between the
      components of the edge feature (h_i, h_j, offset) to determine edge importance.
    - Non-linearity: ReLU activation introduces non-linearity, enabling the MLP to
      model more complex scoring functions than a simple linear combination.
    - Task-specific: The scores can be trained end-to-end to be optimal for the
      downstream task (e.g., guiding attention in a transformer, weighting messages in a GNN).
    """

    def __init__(self, in_dim: int, hidden_dim: int = 128):
        """
        Args:
            in_dim: Dimensionality of the input edge feature (2*D+3).
            hidden_dim: Dimensionality of the hidden layer in the MLP.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),  # First linear layer: maps input edge feature to hidden dim.
            nn.ReLU(inplace=True),  # Activation function. `inplace=True` saves memory.
            nn.Linear(hidden_dim, 1)  # Second linear layer: maps hidden dim to a single scalar score.
        )

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_feats: Tensor of shape (B, V, k, F_in) containing edge features.
        Returns:
            scores: Tensor of shape (B, V, k) with a scalar score for each edge.
        """
        B, V, k_neighbors, F_in = edge_feats.shape

        # --- Reshape for MLP processing ---
        # Purpose: The MLP (nn.Linear) expects input of shape (N_samples, F_in_mlp).
        # We reshape the (B, V, k, F_in) tensor into (B*V*k, F_in) so that each of the
        # B*V*k edge feature vectors is processed independently by the MLP.
        x = edge_feats.reshape(B * V * k_neighbors, F_in)

        scores_flat = self.net(x)  # Shape: (B*V*k, 1)

        # --- Reshape scores back to original edge structure ---
        # Purpose: To restore the (B, V, k) structure for the output scores,
        # corresponding to the input edges.
        # `.view(B, V, k_neighbors)` reshapes the flat scores.
        scores = scores_flat.view(B, V, k_neighbors)
        return scores

