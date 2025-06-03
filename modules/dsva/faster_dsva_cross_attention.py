
import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeScoringMLP(nn.Module):
    def __init__(self, embed_dim, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2 + 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, edge_features):
        """
        edge_features: (N_edges, 2*D + 3)
        returns: (N_edges,) scalar scores per edge
        """
        return self.mlp(edge_features).squeeze(-1)


class SparseDynamicVoxelAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 4,
        knn_size: int = 10,       # Number of neighbors to consider per anchor
        top_k_select: int = 4,    # Number of neighbors to keep after scoring
        edge_dropout_rate: float = 0.1  # Dropout on edge weights
    ):
        super().__init__()
        self.dim = dim
        self.knn_size = knn_size
        self.top_k_select = top_k_select

        self.edge_scorer = EdgeScoringMLP(self.dim)
        self.edge_dropout = nn.Dropout(edge_dropout_rate)
        self.cross_attn = CrossAttention(dim=self.dim, num_heads=num_heads)

    def forward(self, voxel_tokens: torch.Tensor, voxel_coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel_tokens: (B, V, D)
            voxel_coords: (B, V, 3)
            mask:         (B, V)   boolean
        Returns:
            output: Tensor of shape (B, V, D) with updated features (zeros where mask is False)
        """
        B, V, D = voxel_tokens.shape
        device = voxel_tokens.device

        # 1) Gather non-empty indices and lengths
        coords_list = []
        tokens_list = []
        lengths = torch.zeros(B, dtype=torch.long, device=device)

        for b in range(B):
            idx_b = mask[b].nonzero(as_tuple=False).squeeze(-1)  # (V'_b,)
            lengths[b] = idx_b.numel()
            coords_list.append(voxel_coords[b, idx_b])  # (V'_b, 3)
            tokens_list.append(voxel_tokens[b, idx_b])  # (V'_b, D)

        Vp_max = int(lengths.max().item())  # max V' across batch

        # 2) Prepare output tensor (dense) and zero-fill
        output = torch.zeros(B, V, D, device=device)
        if Vp_max == 0:
            return output  # all voxels are empty

        # 3) Build padded coords & tokens: (B, Vp_max, 3) and (B, Vp_max, D)
        padded_coords = torch.zeros(B, Vp_max, 3, device=device)
        padded_tokens = torch.zeros(B, Vp_max, D, device=device)
        for b in range(B):
            L = lengths[b].item()
            if L > 0:
                padded_coords[b, :L] = coords_list[b]
                padded_tokens[b, :L] = tokens_list[b]

        # 4) Compute batched distances: (B, Vp_max, Vp_max)
        dist_batch = torch.cdist(padded_coords, padded_coords, p=2)

        # Mask out invalid rows/cols by setting to +inf
        inf = torch.tensor(float("inf"), device=device)
        for b in range(B):
            L = lengths[b].item()
            if L < Vp_max:
                dist_batch[b, L:, :] = inf
                dist_batch[b, :, L:] = inf

        # 5) For each batch, handle anchors
        #    We will produce attn_out_padded: (B, Vp_max, D)
        attn_out_padded = torch.zeros(B, Vp_max, D, device=device)

        for b in range(B):
            L = lengths[b].item()
            if L == 0:
                continue  # nothing to do
            if L == 1:
                # Only one voxel: no neighbors. Directly copy token.
                attn_out_padded[b, 0] = padded_tokens[b, 0]
                continue

            # Determine effective knn: we need at least (self.knn_size + 1), but if L <= knn_size, clamp
            effective_k = min(self.knn_size + 1, L)
            # 5a) Topk on distances for sample b
            dist_b = dist_batch[b, :L, :L]  # (L, L)
            knn_all = dist_b.topk(effective_k, largest=False, dim=1).indices  # (L, effective_k)
            # Drop self-index: each row's first entry is self (distance=0)
            if effective_k > 1:
                knn_idx_b = knn_all[:, 1:]  # (L, effective_k-1)
            else:
                # effective_k == 1 means only self; so no neighbors
                knn_idx_b = torch.zeros(L, 0, dtype=torch.long, device=device)

            # If we need exactly self.knn_size neighbors, but L-1 < knn_size,
            # pad with zeros so shape=(L, self.knn_size)
            if knn_idx_b.shape[1] < self.knn_size:
                pad_width = self.knn_size - knn_idx_b.shape[1]
                pad_tensor = knn_idx_b.new_zeros(L, pad_width)
                knn_idx_b = torch.cat([knn_idx_b, pad_tensor], dim=1)  # (L, knn_size)
            else:
                # Keep only first knn_size
                knn_idx_b = knn_idx_b[:, : self.knn_size]  # (L, knn_size)

            # 6) Gather neighbor coords & feats for sample b
            anchor_coords_b = padded_coords[b, :L].unsqueeze(1).expand(L, self.knn_size, 3)  # (L, knn_size, 3)
            neighbor_coords_b = padded_coords[b, knn_idx_b]  # (L, knn_size, 3)

            anchor_feats_b = padded_tokens[b, :L].unsqueeze(1).expand(L, self.knn_size, D)  # (L, knn_size, D)
            neighbor_feats_b = padded_tokens[b, knn_idx_b]  # (L, knn_size, D)

            # 7) Relative positions & edge features
            rel_pos_b = neighbor_coords_b - anchor_coords_b  # (L, knn_size, 3)
            edge_feat_b = torch.cat([anchor_feats_b, neighbor_feats_b, rel_pos_b], dim=-1)  # (L, knn_size, 2D+3)

            # 8) Score edges
            edge_feat_flat = edge_feat_b.view(-1, 2 * D + 3)  # (L * knn_size, 2D+3)
            edge_scores_flat = self.edge_scorer(edge_feat_flat)  # (L * knn_size,)
            edge_scores = edge_scores_flat.view(L, self.knn_size)  # (L, knn_size)

            # 9) Attention weights & dropout
            attn_weights = F.softmax(edge_scores, dim=1)  # (L, knn_size)
            attn_weights = self.edge_dropout(attn_weights)

            # 10) Select top_k_select neighbors per anchor
            if self.top_k_select < self.knn_size:
                topk_vals, topk_local_idx = attn_weights.topk(self.top_k_select, dim=1)  # (L, top_k_select)
                topk_abs_idx = knn_idx_b.gather(1, topk_local_idx)  # (L, top_k_select)
                topk_weights = topk_vals / (topk_vals.sum(dim=1, keepdim=True) + 1e-10)  # (L, top_k_select)
            else:
                # top_k_select >= knn_size: keep all
                topk_abs_idx = knn_idx_b  # (L, knn_size)
                topk_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-10)  # (L, knn_size)

            k_final = topk_abs_idx.shape[1]  # = top_k_select or knn_size

            # 11) Gather final neighbor features for cross-attn
            neighbor_feats_final = padded_tokens[b, topk_abs_idx]  # (L, k_final, D)
            anchor_feats_final = padded_tokens[b, :L].unsqueeze(1)  # (L, 1, D)

            # 12) Perform cross-attention for sample b
            #    Need to reshape anchor_feats_final → (1, L, D) and neighbor_feats_final → (1, L, k_final, D)
            attn_input_q = anchor_feats_final.unsqueeze(0)  # (1, L, D)
            attn_input_kv = neighbor_feats_final.unsqueeze(0)  # (1, L, k_final, D)
            attn_out_b = self.cross_attn(attn_input_q, attn_input_kv).squeeze(0)  # (L, D)

            attn_out_padded[b, :L] = attn_out_b  # place in padded output

        # 13) Scatter back into dense output
        for b in range(B):
            idx_b = mask[b].nonzero(as_tuple=False).squeeze(-1)  # (V'_b,)
            L = lengths[b].item()
            if L > 0:
                output[b, idx_b] = attn_out_padded[b, :L]

        return output  # shape (B, V, D)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, anchor_tokens: torch.Tensor, neighbor_tokens: torch.Tensor):
        """
        anchor_tokens:   (1, L, D)
        neighbor_tokens: (1, L, k, D)
        Returns:
            attended: (1, L, D)
        """
        Bq, L, D = anchor_tokens.shape
        k = neighbor_tokens.size(2)
        H = self.num_heads
        Dh = self.head_dim

        # Q: (Bq, H, L, Dh)
        Q = self.q_proj(anchor_tokens).view(Bq, L, H, Dh).transpose(1, 2)

        # K: (Bq, H, L, k, Dh)
        K = self.k_proj(neighbor_tokens).view(Bq, L, k, H, Dh).permute(0, 3, 1, 2, 4)

        # V_: (Bq, H, L, k, Dh)
        V_ = self.v_proj(neighbor_tokens).view(Bq, L, k, H, Dh).permute(0, 3, 1, 2, 4)

        # Attention scores: (Bq, H, L, k)
        attn_scores = (Q.unsqueeze(3) * K).sum(-1) / (Dh ** 0.5)

        # Softmax over k
        attn_weights = F.softmax(attn_scores, dim=-1)  # (Bq, H, L, k)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum: (Bq, H, L, Dh)
        attn_output = (attn_weights.unsqueeze(-1) * V_).sum(dim=3)

        # Reshape → (Bq, L, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(Bq, L, D)

        return self.out_proj(attn_output)


def gather_neighbors_by_indices(tokens: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    tokens:  (B, Vp, D) or (1, Vp, D)
    indices: (1, Vp, k) or (B, Vp, k)
    Returns:
        gathered: (B, Vp, k, D)
    """
    B, Vq, k = indices.shape
    _, Vp, D = tokens.shape
    expanded_indices = indices.unsqueeze(-1).expand(-1, -1, -1, D)
    expanded_tokens = tokens.unsqueeze(1).expand(-1, Vq, -1, -1)
    gathered = torch.gather(expanded_tokens, 2, expanded_indices)
    return gathered
