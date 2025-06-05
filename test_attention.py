# tests/test_dsva_attention.py

import numpy as np
import pytest
import torch
import torch.nn as nn

# Import both versions; adjust paths if needed:
from modules.dsva.dsva_cross_attention import SparseDynamicVoxelAttention as OldSparseDynamicVoxelAttention
from modules.dsva.faster_dsva_cross_attention import SparseDynamicVoxelAttention


def _make_random_input(
    batch_size: int,
    V: int,
    D: int,
    nonempty_frac: float,
    device: torch.device,
    seed: int = None,
):
    """
    Create a random batch of inputs for dynamic sparse voxel attention:
      - tokens:  (B, V, D)  float32
      - coords:  (B, V, 3)  float32
      - mask:    (B, V)     bool, with ~nonempty_frac True
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    tokens = torch.randn(batch_size, V, D, device=device)
    coords = torch.randn(batch_size, V, 3, device=device) * 5.0  # spread out in space

    # Random boolean mask, with approximately `nonempty_frac` fraction of True
    mask = torch.rand(batch_size, V, device=device) < nonempty_frac
    return tokens, coords, mask


@pytest.mark.parametrize("device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"])
@pytest.mark.parametrize("batch_size, V, D, k_knn, k_top, nonempty_frac", [
    # small toy batch
    (1, 10, 4, 3, 1, 0.5),
    # single‐voxel case
    (2, 5, 8, 2, 1, 0.2),
    # moderate size
    (3, 32, 16, 5, 2, 0.6),
    # larger features
    (2, 64, 32, 8, 3, 0.75),
])
def test_old_vs_new_match(batch_size, V, D, k_knn, k_top, nonempty_frac, device):
    """
    For several random seeds and input sizes, check that
    OldSparseDynamicVoxelAttention and SparseDynamicVoxelAttention
    produce nearly identical outputs (within floating‐point tolerance).
    """

    # 1) Create random inputs
    tokens, coords, mask = _make_random_input(
        batch_size=batch_size,
        V=V,
        D=D,
        nonempty_frac=nonempty_frac,
        device=torch.device(device),
        seed=42,
    )

    # 2) Instantiate both modules with the same weights
    old_attn = OldSparseDynamicVoxelAttention(dim=D, knn_size=k_knn, top_k_select=k_top, num_heads=4).to(device)
    new_attn = SparseDynamicVoxelAttention(dim=D, knn_size=k_knn, top_k_select=k_top, num_heads=4).to(device)

    # Copy parameters (edge‐scorer MLP and final linear) from old → new so they match exactly
    def _copy_parameters(src: nn.Module, dst: nn.Module):
        src_state = src.state_dict()
        dst_state = dst.state_dict()
        for k in src_state:
            if k in dst_state and src_state[k].shape == dst_state[k].shape:
                dst_state[k].copy_(src_state[k])

    _copy_parameters(old_attn.edge_scorer, new_attn.edge_scorer)
    # The new implementation has a final Linear(2D→D) created inside forward, so we must
    # re‐create it here with the same weights/offsets. In practice, you should extract that
    # Linear into `self.final_linear = nn.Linear(2*D, D)` in the new code, but for now:
    # assume old_attn had a similar final_linear named `old_attn.final_linear`.
    try:
        # If both classes define a final_linear with identical shapes, copy it
        new_attn_final = new_attn.final_linear
        old_attn_final = old_attn.final_linear
        new_attn_final.weight.data.copy_(old_attn_final.weight.data)
        new_attn_final.bias.data.copy_(old_attn_final.bias.data)
    except AttributeError:
        # If old_attn implemented final linear inside forward (anonymous),
        # we skip copying. In that case the comparison must allow for small differences.
        pass

    old_attn.eval()
    new_attn.eval()

    # 3) Run forward pass on both
    with torch.no_grad():
        out_old = old_attn(tokens, coords, mask)
        out_new = new_attn(tokens, coords, mask)

    # 4) Both outputs should be shaped (B, V, D)
    assert out_old.shape == (batch_size, V, D)
    assert out_new.shape == (batch_size, V, D)

    # 5) Assert that they match closely on the “active” (mask=True) entries;
    #    we expect zero (or identical) entries where mask == False
    #    Use torch.testing.assert_close with a small tol.
    #    We compare out_old and out_new on all entries, but allow small epsilon.
    torch.testing.assert_close(out_new, out_old, rtol=1e-4, atol=1e-6)


def test_all_mask_false(batch_size=2, V=16, D=8, k_knn=4, k_top=2, device="cpu"):
    """
    If mask is all False (no nonempty voxels), both implementations should return all zeros.
    """
    tokens = torch.randn(batch_size, V, D, device=device)
    coords = torch.randn(batch_size, V, 3, device=device)
    mask = torch.zeros(batch_size, V, dtype=torch.bool, device=device)

    # 2) Instantiate both modules with the same weights
    old_attn = OldSparseDynamicVoxelAttention(dim=D, knn_size=k_knn, top_k_select=k_top, num_heads=4).to(device)
    new_attn = SparseDynamicVoxelAttention(dim=D, knn_size=k_knn, top_k_select=k_top, num_heads=4).to(device)


    old_attn.eval()
    new_attn.eval()

    with torch.no_grad():
        out_old = old_attn(tokens, coords, mask)
        out_new = new_attn(tokens, coords, mask)

    # Both should be zeros
    assert torch.allclose(out_old, torch.zeros_like(out_old))
    assert torch.allclose(out_new, torch.zeros_like(out_new))


def test_single_voxel_only(batch_size=3, V=10, D=6, k_knn=3, k_top=1, device="cpu"):
    """
    If each batch sample has exactly one nonempty voxel (mask sum = 1), both implementations
    should simply reproduce the original token at that position (no neighbors exist).
    """
    tokens = torch.randn(batch_size, V, D, device=device)
    coords = torch.randn(batch_size, V, 3, device=device)
    mask = torch.zeros(batch_size, V, dtype=torch.bool, device=device)

    # Force exactly one True per row in mask
    for b in range(batch_size):
        idx = b % V
        mask[b, idx] = True

        # 2) Instantiate both modules with the same weights
        old_attn = OldSparseDynamicVoxelAttention(dim=D, knn_size=k_knn, top_k_select=k_top, num_heads=4).to(device)
        new_attn = SparseDynamicVoxelAttention(dim=D, knn_size=k_knn, top_k_select=k_top, num_heads=4).to(device)

    old_attn.eval()
    new_attn.eval()

    with torch.no_grad():
        out_old = old_attn(tokens, coords, mask)
        out_new = new_attn(tokens, coords, mask)

    # Check: At each b, the only nonzero position in mask is at index idx=b%V.
    for b in range(batch_size):
        idx = b % V
        # out_old[b, idx] should match tokens[b, idx]; same for out_new
        assert torch.allclose(out_old[b, idx], tokens[b, idx], rtol=1e-5, atol=1e-6)
        assert torch.allclose(out_new[b, idx], tokens[b, idx], rtol=1e-5, atol=1e-6)

        # All other positions must be zero
        zeros_old = torch.zeros(D, device=device)
        zeros_new = torch.zeros(D, device=device)
        for v in range(V):
            if v != idx:
                assert torch.allclose(out_old[b, v], zeros_old, atol=1e-6)
                assert torch.allclose(out_new[b, v], zeros_new, atol=1e-6)


@pytest.mark.parametrize("batch_size,V,D,k_knn,k_top", [
    (2, 20, 16, 5, 5),  # edge case: k_top == k_knn
    (2, 20, 16, 1, 1),  # edge case: k_knn = k_top = 1
])
def test_edge_cases_equal_neighbors(batch_size, V, D, k_knn, k_top):
    """
    If k_top == k_knn, then after scoring, we keep all neighbors—thus both implementations
    become deterministic and should match exactly. Similarly, if k_knn = 1 and k_top=1,
    each anchor only attends to itself (distance=0), so output==input for masked positions.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokens, coords, mask = _make_random_input(
        batch_size=batch_size,
        V=V,
        D=D,
        nonempty_frac=0.7,
        device=device,
        seed=7,
    )

    # 2) Instantiate both modules with the same weights
    old_attn = OldSparseDynamicVoxelAttention(dim=D, knn_size=k_knn, top_k_select=k_top, num_heads=4).to(device)
    new_attn = SparseDynamicVoxelAttention(dim=D, knn_size=k_knn, top_k_select=k_top, num_heads=4).to(device)

    # Copy edge_scorer/ final linear as before
    _copy_parameters = lambda src, dst: dst.load_state_dict(src.state_dict(), strict=False)
    try:
        _copy_parameters(old_attn.edge_scorer, new_attn.edge_scorer)
        _copy_parameters(old_attn.final_linear, new_attn.final_linear)
    except Exception:
        pass

    old_attn.eval()
    new_attn.eval()

    with torch.no_grad():
        out_old = old_attn(tokens, coords, mask)
        out_new = new_attn(tokens, coords, mask)

    # Verify they match exactly on all entries
    torch.testing.assert_close(out_new, out_old, atol=1e-6, rtol=1e-5)


def test_gradients_agree_small(batch_size=1, V=12, D=8, k_knn=4, k_top=2, device="cpu"):
    """
    A small test to check that backward gradients w.r.t. `tokens` and `coords` match
    (approximately) between old and new. We use finite differences / gradcheck‐style.
    Because `coords` enter through `torch.cdist`, which is non‐differentiable wrt coords in PyTorch,
    we only check gradients w.r.t. `tokens` here.
    """
    tokens, coords, mask = _make_random_input(
        batch_size=batch_size, V=V, D=D, nonempty_frac=0.5,
        device=torch.device(device), seed=123
    )
    tokens.requires_grad_(True)

    # 2) Instantiate both modules with the same weights
    old_attn = OldSparseDynamicVoxelAttention(dim=D, knn_size=k_knn, top_k_select=k_top, num_heads=4).to(device)
    new_attn = SparseDynamicVoxelAttention(dim=D, knn_size=k_knn, top_k_select=k_top, num_heads=4).to(device)

    # Copy weights
    _copy_parameters = lambda src, dst: dst.load_state_dict(src.state_dict(), strict=False)
    try:
        _copy_parameters(old_attn.edge_scorer, new_attn.edge_scorer)
        _copy_parameters(old_attn.final_linear, new_attn.final_linear)
    except Exception:
        pass

    old_attn.train()
    new_attn.train()

    # Forward + backward on old
    out_old = old_attn(tokens, coords, mask)          # (1, V, D)
    loss_old = out_old.sum()
    loss_old.backward()
    grad_tokens_old = tokens.grad.clone()
    tokens.grad.zero_()

    # Forward + backward on new
    out_new = new_attn(tokens, coords, mask)
    loss_new = out_new.sum()
    loss_new.backward()
    grad_tokens_new = tokens.grad.clone()

    # The gradients should match (up to floating‐point differences)
    torch.testing.assert_close(grad_tokens_new, grad_tokens_old, rtol=1e-3, atol=1e-4)


if __name__ == "__main__":
    # Run tests if this file is executed directly
    pytest.main([__file__])
