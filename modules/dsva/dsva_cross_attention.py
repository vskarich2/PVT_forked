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
        edge_scores = self.mlp(edge_features).squeeze(-1)
        return edge_scores

class SparseDynamicVoxelAttention(nn.Module):
    def __init__(
            self,
            dim = None,
            num_heads=4,
            knn_size=10, # Number of neighbors in a neighborhood to consider
            top_k_select=4 # A top ranked subset extracted from knn_size to use for cross-attention
    ):
        super().__init__()
        self.dim = dim
        self.knn_size = knn_size
        self.top_k_select = top_k_select

        self.edge_scorer = EdgeScoringMLP(self.dim)
        self.edge_dropout = nn.Dropout(0.1)
        self.cross_attn = CrossAttention(dim=self.dim, num_heads=num_heads)

    def forward(self, voxel_tokens, voxel_coords, mask):
        """
        Args:
            voxel_tokens: Tensor of shape (B, V, D)
                • B = batch size
                • V = total number of voxels (grid cells)
                • D = token embedding dimension

            voxel_coords: Tensor of shape (B, V, 3)
                • the (x,y,z) center of each voxel

            mask: BoolTensor of shape (B, V)
                • True where the voxel is non-empty

        Returns:
            updated_tokens: List of length B; each entry is a Tensor of shape (Vʼ, D)
                • Vʼ = number of non-empty voxels in that batch element
        """
        B, V, D = voxel_tokens.shape
        updated_tokens = []  # will collect one (Vʼ, D) per batch

        # Note this for loop is not efficient. TODO: Vectorize Batch Loop
        for b in range(B):
            # ——— 1) Select only non-empty voxels for sample b ————————————————
            # mask[b] is (V,) bool → indexing drops empty voxels
            token = voxel_tokens[b][mask[b]]  # → (Vʼ, D)
            coord = voxel_coords[b][mask[b]]  # → (Vʼ, 3)
            Vp = token.size(0)  # Vʼ = number of non-empty voxels

            # ——— 2) Compute all-pairs distances among the Vʼ voxel centers ————
            # unsqueeze to (1, Vʼ, 3), so cdist returns (1, Vʼ, Vʼ)
            dist = torch.cdist(
                coord.unsqueeze(0),
                coord.unsqueeze(0),
                p=2
            )  # → (1, Vʼ, Vʼ)

            # ——— 3) Get k_knn nearest *other* voxels for each anchor ————————
            # +1 because the nearest neighbor of each point is itself (dist=0)
            # .indices gives shape (1, Vʼ, k_knn+1), then we drop index 0
            knn_idx = dist.topk(
                k=self.knn_size + 1,
                dim=-1,
                largest=False
            ).indices[:, :, 1:]  # → (1, Vʼ, k_knn)

            # ——— 4) Gather the neighbor tokens & coords ————————————————

            # Helper that does for you: x_neighbors[i,j,:] = x[j, knn_idx[i,j]]
            knn_tokens = gather_neighbors_by_indices(
                token.unsqueeze(0),  # (1, Vʼ, D)
                knn_idx  # (1, Vʼ, k_knn)
            )  # → (1, Vʼ, k_knn, D)

            knn_coords = gather_neighbors_by_indices(
                coord.unsqueeze(0),  # (1, Vʼ, 3)
                knn_idx
            )  # → (1, Vʼ, k_knn, 3)

            # ——— 5) Build “anchor” tensors for pairing & compute relative pos ——

            # replicate each anchor token/coord k_knn times to align with knn_tokens
            anchor_tokens = token.unsqueeze(0) \
                .unsqueeze(2) \
                .expand(-1, Vp, self.knn_size, -1)

            # → (1, Vʼ, k_knn, D)
            anchor_coords = coord.unsqueeze(0) \
                .unsqueeze(2) \
                .expand(-1, Vp, self.knn_size, -1)
            # → (1, Vʼ, k_knn, 3)
            # relative position vector for edge features
            rel_pos = knn_coords - anchor_coords  # → (1, Vʼ, k_knn, 3)

            # ——— 6) Score each “edge” with an MLP ————————————————
            # concat [anchor_token, neighbor_token, relative_pos]
            edge_feats = torch.cat(
                [anchor_tokens, knn_tokens, rel_pos],
                dim=-1
            )  # → (1, Vʼ, k_knn, 2D+3)

            # collapse the last MLP dim to a scalar per edge
            edge_scores = self.edge_scorer(edge_feats) \
                .squeeze(-1)  # → (1, Vʼ, k_knn)

            attn_weights = F.softmax(edge_scores, dim=-1)
            attn_weights = self.edge_dropout(attn_weights)

            # ——— 7) Pick the top-k_select strongest edges ————————————
            topk_scores, topk_local_idx = attn_weights.topk(
                self.top_k_select,
                dim=-1
            )  # both (1, Vʼ, k_select)

            # map from local position in knn_idx to the original voxel index
            topk_indices = torch.gather(
                knn_idx,  # (1, Vʼ, k_knn)
                2,
                topk_local_idx  # (1, Vʼ, k_select)
            )  # → (1, Vʼ, k_select)

            # ——— 8) Gather the selected neighbor tokens for attention ————
            topk_neighbors = gather_neighbors_by_indices(
                token.unsqueeze(0),  # (1, Vʼ, D)
                topk_indices  # (1, Vʼ, k_select)
            )  # → (1, Vʼ, k_select, D)

            # ——— 9) Cross-attention: each anchor attends over its top-k neighbors —
            # query = (1, Vʼ, D), keys/values = (1, Vʼ, k_select, D)
            out = self.cross_attn(
                token.unsqueeze(0),  # queries (1, Vʼ, D)
                topk_neighbors  # key/value (1, Vʼ, k_select, D)
            )  # → (1, Vʼ, D)

            # store the result for this batch member
            updated_tokens.append(out.squeeze(0))  # → (Vʼ, D)

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
        self.attn_dropout = nn.Dropout(0.1)

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

