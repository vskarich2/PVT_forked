import warnings
import os
import torch
import traceback
from modules.functional.python_fallback import *

try:
    from torch.utils.cpp_extension import load
    _src_path = os.path.dirname(os.path.abspath(__file__))
    _backend = load(
        name='_pvt_backend',
        extra_cflags=['-O3', '-std=c++17'],
        sources=[os.path.join(_src_path, 'src', f) for f in [
            'interpolate/neighbor_interpolate.cpp',
            'interpolate/neighbor_interpolate.cu',
            'interpolate/trilinear_devox.cpp',
            'interpolate/trilinear_devox.cu',
            'sampling/sampling.cpp',
            'sampling/sampling.cu',
            'voxelization/vox.cpp',
            'voxelization/vox.cu',
            'bindings.cpp',
        ]]
    )
except Exception as e:
    # If CUDA build fails, fall back to CPU stubs.
    from modules.functional.python_fallback import *

    class CPUBackend:
        @staticmethod
        def trilinear_devoxelize_forward(
                resolution: int,
                is_training: bool,
                coords: torch.Tensor,  # expects (B, N, 3)
                grid: torch.Tensor     # shape (B, C, R, R, R)
        ):
            # 1) Pure‐Python interpolation
            outs = trilinear_devoxelize_cpu(
                grid=grid,
                coords=coords,      # must be (B, N, 3)
                resolution=resolution
            )  # returns (B, C, N)

            # 2) Build a flat‐index map for backward
            idx_int = coords.long()  # (B, N, 3)
            idx_flat = (
                idx_int[..., 0] * (resolution * resolution)
                + idx_int[..., 1] * resolution
                + idx_int[..., 2]
            )  # shape (B, N)

            # 3) Create dummy weights (used only for gradient routing)
            B, C, N = outs.shape
            wgts = torch.ones(B, C, N, device=outs.device, dtype=outs.dtype)

            return outs, idx_flat, wgts  # all three outputs

        @staticmethod
        def trilinear_devoxelize_backward(
                resolution: int,
                is_training: bool,
                coords: torch.Tensor,   # shape (B, N, 3)
                grid: torch.Tensor,     # placeholder, not used
                grad_out: torch.Tensor  # shape (B, C, R, R, R)
        ):
            # Gather gradient at nearest grid cell
            B, C, R1, R2, R3 = grad_out.shape
            R = R1
            idx_int = coords.long()  # (B, N, 3)
            idx_flat = (
                idx_int[..., 0] * (R * R) +
                idx_int[..., 1] * R +
                idx_int[..., 2]
            )  # (B, N)

            grad_flat = grad_out.view(B, C, -1)  # (B, C, R³)
            grad_feats = torch.gather(
                grad_flat, 2,
                idx_flat.unsqueeze(1).expand(-1, C, -1)
            )  # (B, C, N)
            return grad_feats  # (B, C, N)

        @staticmethod
        def avg_voxelize_forward(features, coords, resolution):
            return avg_voxelize_forward_cpu(features, coords, resolution)

        @staticmethod
        def avg_voxelize_backward(grad_out, flat_idx, counts):
            return avg_voxelize_backward_cpu(grad_out, flat_idx, counts)

        @staticmethod
        def sparse_window_attention(feats, window_size, num_heads, head_dim, **kw):
            return sparse_window_attention_cpu(feats, window_size, num_heads, head_dim)

        @staticmethod
        def trilinear_devoxelize(grid, coords, resolution, training=False, **kw):
            return trilinear_devoxelize_cpu(grid, coords, resolution, training=training)

    _backend = CPUBackend()

__all__ = ['_backend']
