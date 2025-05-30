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

    def forward(self, anchor_tokens, neighbor_tokens, rel_pos):
        edge_features = torch.cat([anchor_tokens, neighbor_tokens, rel_pos], dim=-1)
        edge_scores = self.mlp(edge_features).squeeze(-1)
        return edge_scores


class SparseDynamicVoxelAttention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=4,
            k_knn=10,
            k_select=4,
            edge_hidden_dim=64
    ):
        super().__init__()

        self.k_knn = k_knn
        self.k_select = k_select

        self.edge_scorer = EdgeScoringMLP(dim)
        self.cross_attn = CrossAttention(dim=dim, num_heads=num_heads)

    def forward(self, voxel_tokens, voxel_coords, mask):
        """
        Args:
            voxel_tokens: (B, V, D) - token embeddings
            voxel_coords: (B, V, 3) - voxel centers
            mask: (B, V) - boolean tensor for non-empty voxels
        Returns:
            updated_tokens: list of (V', D) tensors, one per batch
        """
        B, V, D = voxel_tokens.shape
        updated_tokens = []

        for b in range(B):
            token = voxel_tokens[b][mask[b]]  # (V', D)
            coord = voxel_coords[b][mask[b]]  # (V', 3)
            Vp = token.size(0)

            # Compute distances and kNN indices
            dist = torch.cdist(coord.unsqueeze(0), coord.unsqueeze(0), p=2)  # (1, V', V')
            knn_idx = dist.topk(k=self.k_kn + 1, dim=-1, largest=False).indices[:, :, 1:]  # (1, V', k_knn)

            # Gather neighbor tokens and coords
            knn_tokens = gather_neighbors_by_indices(token.unsqueeze(0), knn_idx)  # (1, V', k, D)
            knn_coords = gather_neighbors_by_indices(coord.unsqueeze(0), knn_idx)  # (1, V', k, 3)
            anchor_tokens = token.unsqueeze(0).unsqueeze(2).expand(-1, Vp, self.k_kn, -1)
            anchor_coords = coord.unsqueeze(0).unsqueeze(2).expand(-1, Vp, self.k_kn, -1)
            rel_pos = knn_coords - anchor_coords  # (1, V', k, 3)

            # Score edges
            edge_feats = torch.cat([anchor_tokens, knn_tokens, rel_pos], dim=-1)  # (1, V', k, 2D+3)
            edge_scores = self.edge_scorer(edge_feats).squeeze(-1)  # (1, V', k)

            # Top-k select
            topk_scores, topk_local_idx = edge_scores.topk(self.k_select, dim=-1)
            topk_indices = torch.gather(knn_idx, 2, topk_local_idx)  # (1, V', k_select)

            # Gather top-k neighbor tokens
            topk_neighbors = gather_neighbors_by_indices(token.unsqueeze(0), topk_indices)  # (1, V', k_select, D)

            # Cross-attention
            out = self.cross_attn(token.unsqueeze(0), topk_neighbors)  # (1, V', D)
            updated_tokens.append(out.squeeze(0))  # (V', D)

        return updated_tokens

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, anchor_tokens, neighbor_tokens):
        B, V, D = anchor_tokens.shape
        k = neighbor_tokens.size(2)
        H = self.num_heads
        Dh = self.head_dim
        Q = self.q_proj(anchor_tokens).view(B, V, H, Dh).transpose(1, 2)
        K = self.k_proj(neighbor_tokens).view(B, V, k, H, Dh).permute(0, 3, 1, 2, 4)
        V_ = self.v_proj(neighbor_tokens).view(B, V, k, H, Dh).permute(0, 3, 1, 2, 4)
        attn_scores = (Q.unsqueeze(3) * K).sum(-1) / Dh**0.5
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_output = (attn_weights.unsqueeze(-1) * V_).sum(dim=3)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, V, D)
        return self.out_proj(attn_output)


def gather_neighbors_by_indices(tokens, indices):
    B, V, D = tokens.shape
    B2, Vq, k = indices.shape
    assert B == B2
    expanded_indices = indices.unsqueeze(-1).expand(-1, -1, -1, D)
    expanded_tokens = tokens.unsqueeze(1).expand(-1, Vq, -1, -1)
    gathered = torch.gather(expanded_tokens, 2, expanded_indices)
    return gathered


# # Example usage with dummy data
# B, R, D, V = 2, 8, 64, 512
# tokens = torch.randn(B, V, D)
# coords = torch.randn(B, V, 3)
# mask = torch.rand(B, V) > 0.3  # simulate ~70% non-empty voxels
#
# sparse_attn = SparseDynamicVoxelAttention(dim=D)
# updated = sparse_attn(tokens, coords, mask)

