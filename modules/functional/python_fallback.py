# python_fallback.py
import torch.nn.functional as F
import math
import torch
def trilinear_devoxelize_forward_cpu(features, coords, resolution):
    """
    CPU stub matching signature:
      inputs: features (B, C, N), coords (B, N, 3), resolution
      outputs: outs (B, C, R, R, R), inds (B, N), wgts (B, C, N)
    """
    # reuse your existing trilinear_devoxelize_cpu logic, but return also inds & weights
    outs = trilinear_devoxelize_cpu(
        grid=features.view(features.size(0), features.size(1), resolution, resolution, resolution),
        coords=coords,
        resolution=resolution,
        training=False
    )
    # python_fallback.trilinear_devoxelize_cpu returns (B, C, N)
    # we need outs, plus dummy inds & wgts:
    #   inds = flat_idx from avg_voxelize_forward (not used later)
    #   wgts = the trilinear weights per point-channel (we can return ones)
    B, C, N = features.size()
    device, dtype = features.device, features.dtype
    inds = torch.zeros(B, N, dtype=torch.long, device=device)
    wgts = torch.ones(B, C, N, dtype=dtype, device=device)
    return outs, inds, wgts


def trilinear_devoxelize_backward_cpu(grad_out, coords, resolution):
    """
    CPU stub matching signature:
      inputs: grad_out (B, C, R, R, R), coords (B, N, 3), resolution
      output: grad_features (B, C, N)
    """
    # Simplest: distribute gradient equally (or just gather)
    # Here we just gather the gradient from the nearest voxel:
    B, C, R1, R2, R3 = grad_out.size()
    R = R1
    _, N, _ = coords.size()
    # compute flat voxel indices
    idx = coords.long()
    flat_idx = idx[...,0]*R*R + idx[...,1]*R + idx[...,2]  # (B, N)
    grad_flat = grad_out.view(B, C, -1)
    # gather
    grad_feats = torch.gather(
        grad_flat, 2,
        flat_idx.unsqueeze(1).expand(-1, C, -1)
    )
    return grad_feats

def avg_voxelize_forward_cpu(
    features: torch.Tensor,
    coords: torch.Tensor,
    resolution: int
):
    """
    CPU stub for avg_voxelize_forward:
      - features: (B, C, N) tensor of per-point feature vectors
      - coords:   (B, 3, N) integer voxel indices in [0, resolution-1]
      - resolution: int R, number of voxels along each axis

    Returns:
      out:     (B, C, R, R, R) averaged voxel features
      flat_idx:(B, N) flat voxel index for each point
      counts:  (B, R**3) number of points per voxel
    """
    # unpack batch, channels, points
    B, C, N = features.shape
    R = resolution
    device = features.device
    dtype = features.dtype

    # --- step 1: turn 3D coords into a single flat index per point ---
    # coords comes in as (B, 3, N): first channel=x, second=y, third=z
    x_idx = coords[:, 0, :].long()  # (B, N)  x-coordinate per point
    y_idx = coords[:, 1, :].long()  # (B, N)  y-coordinate per point
    z_idx = coords[:, 2, :].long()  # (B, N)  z-coordinate per point

    # compute flat indices: flat = x*R*R + y*R + z
    flat_idx = x_idx * (R * R) + y_idx * R + z_idx  # (B, N)

    # --- step 2: allocate accumulators for sum and counts ---
    # out_flat will accumulate feature sums per voxel slot
    out_flat = torch.zeros(B, C, R**3, device=device, dtype=dtype)
    # counts will track how many points fell into each voxel (use integer type)
    counts = torch.zeros(B, R**3, device=device, dtype=torch.int64)

    # --- step 3: scatter-add features & count points per batch ---
    for b in range(B):
        fi = flat_idx[b]        # (N,) flat index for batch b
        fv = features[b]        # (C, N) features for batch b

        # expand indices to match feature dims: (C, N)
        fi_exp = fi.unsqueeze(0).expand(C, -1)
        # scatter-add: for each point n, add fv[:,n] into out_flat[b,:,fi[n]]
        out_flat[b].scatter_add_(1, fi_exp, fv)

        # count number of points per flat voxel index
        counts[b] = torch.bincount(fi, minlength=R**3)

    # --- step 4: divide sums by counts to get averages ---
    # create mask of which voxels actually received any points
    nonzero = counts > 0           # (B, R^3) boolean mask
    # convert counts to float for division
    counts_f = counts.to(dtype)    # (B, R^3)

    for b in range(B):
        nz = nonzero[b]             # 1D mask for batch b
        # only divide those voxel slots that are non-empty
        out_flat[b, :, nz] /= counts_f[b, nz]

    # --- step 5: reshape back to 3D voxel grid ---
    # from (B, C, R^3) → (B, C, R, R, R)
    out = out_flat.view(B, C, R, R, R)

    return out, flat_idx, counts


def avg_voxelize_backward_cpu(
    grad_out: torch.Tensor,
    flat_idx: torch.Tensor,
    counts: torch.Tensor
):
    """
    CPU stub for avg_voxelize_backward:
      - grad_out: (B, C, R, R, R)
      - flat_idx: (B, N)
      - counts:   (B, R**3)

    Returns:
      grad_features: (B, C, N)
      None, None     # for coords and resolution args
    """
    B, C, R1, R2, R3 = grad_out.shape
    assert R1 == R2 == R3
    R = R1
    N = flat_idx.shape[1]

    # flatten grad_out to (B, C, R^3)
    grad_flat = grad_out.view(B, C, -1)
    grad_features = torch.zeros(B, C, N, device=grad_out.device, dtype=grad_out.dtype)

    for b in range(B):
        fi = flat_idx[b]                 # (N,)
        cnt = counts[b]                  # (R^3,)
        inv = torch.zeros_like(cnt)
        nz  = cnt > 0
        inv[nz] = 1.0 / cnt[nz]
        # broadcast inv over channel dim
        inv_exp = inv.unsqueeze(0)       # (1, R^3)
        # gather the gradient back to each point
        # grad_flat[b]: (C, R^3)
        # fi.unsqueeze(0).expand(C,N): (C, N)
        grad_features[b] = grad_flat[b].mul(inv_exp).gather(1, fi.unsqueeze(0).expand(C, -1))

    # gradient only w.r.t. features
    return grad_features, None, None

def trilinear_devoxelize_cpu(grid, coords, resolution: int, training: bool = False, **kw):
    """
    CPU fallback for trilinear_devoxelize.

    Args:
        grid:    (B, C, R, R, R) tensor of voxel features.
        coords:  (B, 3, N) float coordinates in [0, R-1] for N points.
        resolution: int R, the grid size.
        training:   ignored here (for API compatibility).
        **kw:        ignored extra args.

    Returns:
        pts_feat: (B, C, N) trilinearly interpolated features at each coord.
    """
    # (B, 3, N) => (B, N, 3)
    coords = coords.permute(0, 2, 1)

    B, C, R = grid.shape
    _, N, _ = coords.shape

    # Flatten spatial dims for easy gather:
    grid_flat = grid.view(B, C, -1)  # (B, C, R^3)

    # Clamp coords just in case:
    coords_clamped = coords.clamp(0, R - 1)

    # Split into x, y, z components:
    x, y, z = coords_clamped.unbind(-1)  # each (B, N)

    # Floor / ceil indices:
    x0 = x.floor().long()
    y0 = y.floor().long()
    z0 = z.floor().long()
    x1 = (x0 + 1).clamp(max=R - 1)
    y1 = (y0 + 1).clamp(max=R - 1)
    z1 = (z0 + 1).clamp(max=R - 1)

    # Compute fractional weights:
    xd = (x - x0.float()).unsqueeze(1)  # (B, 1, N)
    yd = (y - y0.float()).unsqueeze(1)
    zd = (z - z0.float()).unsqueeze(1)

    # Helper to compute flat indices:
    def _flat_idx(xi, yi, zi):
        return xi * (R * R) + yi * R + zi  # (B, N)

    # For each corner, gather features:
    idx000 = _flat_idx(x0, y0, z0)
    idx001 = _flat_idx(x0, y0, z1)
    idx010 = _flat_idx(x0, y1, z0)
    idx011 = _flat_idx(x0, y1, z1)
    idx100 = _flat_idx(x1, y0, z0)
    idx101 = _flat_idx(x1, y0, z1)
    idx110 = _flat_idx(x1, y1, z0)
    idx111 = _flat_idx(x1, y1, z1)

    # Gather corner values: shape each (B, C, N)
    def _gather(idx):
        # idx: (B, N) → expand to (B, C, N) for gather dim=2
        return torch.gather(
            grid_flat, 2,
            idx.unsqueeze(1).expand(-1, C, -1)
        )

    v000 = _gather(idx000)
    v001 = _gather(idx001)
    v010 = _gather(idx010)
    v011 = _gather(idx011)
    v100 = _gather(idx100)
    v101 = _gather(idx101)
    v110 = _gather(idx110)
    v111 = _gather(idx111)

    # Compute interpolation weights:
    w000 = (1 - xd) * (1 - yd) * (1 - zd)
    w001 = (1 - xd) * (1 - yd) * zd
    w010 = (1 - xd) * yd * (1 - zd)
    w011 = (1 - xd) * yd * zd
    w100 = xd * (1 - yd) * (1 - zd)
    w101 = xd * (1 - yd) * zd
    w110 = xd * yd * (1 - zd)
    w111 = xd * yd * zd

    # Weighted sum of corner features:
    pts_feat = (
            v000 * w000 +
            v001 * w001 +
            v010 * w010 +
            v011 * w011 +
            v100 * w100 +
            v101 * w101 +
            v110 * w110 +
            v111 * w111
    )  # (B, C, N)

    return pts_feat


def sparse_window_attention_cpu(
    voxel_feats: torch.Tensor,
    window_size: int,
    num_heads: int,
    head_dim: int,
    **kwargs
) -> torch.Tensor:
    """
    Pure‐Python fallback for PVT’s sparse_window_attention.

    Args:
      voxel_feats: (B, N, C) where C = num_heads * head_dim
      window_size: ignored in this stub
      num_heads: number of attention heads
      head_dim:   dimension per head
      **kwargs:   any other args from the real stub

    Returns:
      Tensor of shape (B, N, C): the “attended” features (here full attention)
    """
    B, N, C = voxel_feats.shape
    assert C == num_heads * head_dim, "C must equal num_heads*head_dim"

    # 1) split into heads
    x = voxel_feats.view(B, N, num_heads, head_dim)            # (B, N, H, d_h)
    x = x.permute(0, 2, 1, 3)                                  # (B, H, N, d_h)

    # 2) compute Q, K, V (here identity projections)
    Q = x  # (B, H, N, d_h)
    K = x  # (B, H, N, d_h)
    V = x  # (B, H, N, d_h)

    # 3) dot‐product attention scores
    #    (B, H, N, d_h) @ (B, H, d_h, N) -> (B, H, N, N)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    attn   = F.softmax(scores, dim=-1)                         # (B, H, N, N)

    # 4) weighted sum
    out = torch.matmul(attn, V)                                # (B, H, N, d_h)

    # 5) merge heads back
    out = out.permute(0, 2, 1, 3).contiguous()                 # (B, N, H, d_h)
    out = out.view(B, N, C)                                    # (B, N, C)

    return out

