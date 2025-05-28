# python_fallback.py
import torch.nn.functional as F
import math
import torch
def trilinear_devoxelize_forward_cpu(features, coords, resolution):
    """
    # Takes voxelized features and maps them back to point cloud coordinates
    # Input shapes:
    # - features: (B, C, N) - batch of N points with C channels
    # - coords: (B, N, 3) - 3D coordinates for each point
    # - resolution: integer size of voxel grid
    """
    # Reshape features to 3D grid and perform devoxelization
    outs = trilinear_devoxelize_cpu(
        grid=features.view(features.size(0), features.size(1), resolution, resolution, resolution),
        coords=coords,
        resolution=resolution,
        training=False
    )
    
    # Create dummy indices and weights (used in the CUDA version but not needed here)
    B, C, N = features.size()
    device, dtype = features.device, features.dtype
    inds = torch.zeros(B, N, dtype=torch.long, device=device)
    wgts = torch.ones(B, C, N, dtype=dtype, device=device)
    return outs, inds, wgts


def trilinear_devoxelize_backward_cpu(grad_out, coords, resolution):
    """
    # Computes gradients for the devoxelization operation
    # Uses a simple nearest-neighbor approach for gradient propagation
    """
    B, C, R1, R2, R3 = grad_out.size()
    R = R1
    _, N, _ = coords.size()
    
    # Convert 3D coordinates to flat indices
    idx = coords.long()
    flat_idx = idx[...,0]*R*R + idx[...,1]*R + idx[...,2]  # (B, N)
    
    # Flatten gradients and gather them using the indices
    grad_flat = grad_out.view(B, C, -1)
    grad_feats = torch.gather(
        grad_flat, 2,
        flat_idx.unsqueeze(1).expand(-1, C, -1)
    )
    return grad_feats

def avg_voxelize_forward_cpu(features, coords, resolution):
    """
    # Converts point cloud to voxel grid by averaging features of points in each voxel
    # This is the forward pass of voxelization
    """
    B, C, N = features.shape
    R = resolution
    device = features.device
    dtype = features.dtype

    # Convert 3D coordinates to flat indices
    idx = coords.long()
    flat_idx = idx[...,0]*R*R + idx[...,1]*R + idx[...,2]

    # Initialize output tensors
    out_flat = torch.zeros(B, C, R**3, device=device, dtype=dtype)
    counts = torch.zeros(B, R**3, device=device, dtype=dtype)

    # Process each batch
    for b in range(B):
        fi = flat_idx[b]               # indices for this batch
        fv = features[b]               # features for this batch
        
        # Sum features into voxels
        fi_exp = fi.unsqueeze(0).expand(C, -1)
        out_flat[b].scatter_add_(1, fi_exp, fv)
        
        # Count points per voxel
        counts[b] = torch.bincount(fi, minlength=R**3)

    # Average features by dividing by count (avoiding divide by zero)
    nonzero = counts > 0
    for b in range(B):
        nz = nonzero[b]
        out_flat[b, :, nz] /= counts[b, nz]

    # Reshape to 3D grid
    out = out_flat.view(B, C, R, R, R)
    return out, flat_idx, counts

def avg_voxelize_backward_cpu(
    grad_out: torch.Tensor,
    flat_idx: torch.Tensor,
    counts: torch.Tensor
):
    """
    # Computes gradients for the voxelization operation
    # Distributes gradients back to the original points
    """
    B, C, R1, R2, R3 = grad_out.shape
    R = R1
    N = flat_idx.shape[1]

    grad_flat = grad_out.view(B, C, -1)
    grad_features = torch.zeros(B, C, N, device=grad_out.device, dtype=grad_out.dtype)

    # Process each batch
    for b in range(B):
        fi = flat_idx[b]
        cnt = counts[b]
        
        # Compute inverse counts (avoiding divide by zero)
        inv = torch.zeros_like(cnt)
        nz = cnt > 0
        inv[nz] = 1.0 / cnt[nz]
        
        # Distribute gradients back to points
        inv_exp = inv.unsqueeze(0)
        grad_features[b] = grad_flat[b].mul(inv_exp).gather(1, fi.unsqueeze(0).expand(C, -1))

    return grad_features, None, None

def trilinear_devoxelize_cpu(grid, coords, resolution, training=False, **kw):
    """
    # Performs trilinear interpolation to map voxel features back to points
    # This is a more sophisticated version of devoxelization that uses
    # interpolation between neighboring voxels
    """
    # Rearrange dimensions for processing
    coords = coords.permute(0, 2, 1)
    B, C, R = grid.shape
    _, N, _ = coords.shape
    grid_flat = grid.view(B, C, -1)

    # Clamp coordinates to valid range
    coords_clamped = coords.clamp(0, R - 1)
    x, y, z = coords_clamped.unbind(-1)

    # Compute floor and ceiling indices for each dimension
    x0 = x.floor().long()
    y0 = y.floor().long()
    z0 = z.floor().long()
    x1 = (x0 + 1).clamp(max=R - 1)
    y1 = (y0 + 1).clamp(max=R - 1)
    z1 = (z0 + 1).clamp(max=R - 1)

    # Compute interpolation weights
    xd = (x - x0.float()).unsqueeze(1)
    yd = (y - y0.float()).unsqueeze(1)
    zd = (z - z0.float()).unsqueeze(1)

    # Helper function for computing flat indices
    def _flat_idx(xi, yi, zi):
        return xi * (R * R) + yi * R + zi

    # Compute indices for all 8 corners of the cube
    idx000 = _flat_idx(x0, y0, z0)
    idx001 = _flat_idx(x0, y0, z1)
    idx010 = _flat_idx(x0, y1, z0)
    idx011 = _flat_idx(x0, y1, z1)
    idx100 = _flat_idx(x1, y0, z0)
    idx101 = _flat_idx(x1, y0, z1)
    idx110 = _flat_idx(x1, y1, z0)
    idx111 = _flat_idx(x1, y1, z1)

    # Gather features from all 8 corners
    def _gather(idx):
        return torch.gather(
            grid_flat, 2,
            idx.unsqueeze(1).expand(-1, C, -1)
        )

    # Get values at all 8 corners
    v000 = _gather(idx000)
    v001 = _gather(idx001)
    v010 = _gather(idx010)
    v011 = _gather(idx011)
    v100 = _gather(idx100)
    v101 = _gather(idx101)
    v110 = _gather(idx110)
    v111 = _gather(idx111)

    # Compute interpolation weights for all 8 corners
    w000 = (1 - xd) * (1 - yd) * (1 - zd)
    w001 = (1 - xd) * (1 - yd) * zd
    w010 = (1 - xd) * yd * (1 - zd)
    w011 = (1 - xd) * yd * zd
    w100 = xd * (1 - yd) * (1 - zd)
    w101 = xd * (1 - yd) * zd
    w110 = xd * yd * (1 - zd)
    w111 = xd * yd * zd

    # Perform trilinear interpolation
    pts_feat = (
        v000 * w000 + v001 * w001 + v010 * w010 + v011 * w011 +
        v100 * w100 + v101 * w101 + v110 * w110 + v111 * w111
    )

    return pts_feat


def sparse_window_attention_cpu(voxel_feats, window_size, num_heads, head_dim, **kwargs):
    """
    # Implements a CPU version of the attention mechanism used in the PVT
    # This is a simplified version that computes full attention rather than
    # the sparse window attention used in the GPU version
    """
    B, N, C = voxel_feats.shape
    
    # Reshape input for multi-head attention
    x = voxel_feats.view(B, N, num_heads, head_dim)
    x = x.permute(0, 2, 1, 3)
    
    # Simple self-attention implementation
    Q = K = V = x  # Identity projections
    
    # Compute attention scores and weights
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    attn = F.softmax(scores, dim=-1)
    
    # Apply attention weights
    out = torch.matmul(attn, V)
    
    # Reshape back to original format
    out = out.permute(0, 2, 1, 3).contiguous()
    out = out.view(B, N, C)
    
    return out

