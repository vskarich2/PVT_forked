import os

try:
    from torch.utils.cpp_extension import load
    _src_path = os.path.dirname(os.path.abspath(__file__))
    _backend = load(name='_pvt_backend',
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
    #error_msg = traceback.format_exc()
    #print(f"Could not build CUDA backend. \nFalling back to CPU stubs.")
    #print(error_msg)

    from modules.functional.python_fallback import *

    class CPUBackend:

        @staticmethod
        def trilinear_devoxelize_forward(
                resolution: int,
                is_training: bool,
                coords: torch.Tensor,  # (B, N, 3) float
                grid: torch.Tensor  # (B, C, R, R, R)
        ):
            # # call your pure‐python interp
            # print("Checking voxel_coords ranges:")
            # print("  min coords:", coords.min(dim=1).values.min(),
            #       coords.min(dim=1).values.min(dim=1).values)
            # print("  max coords:", coords.max(dim=1).values.max(),
            #       coords.max(dim=1).values.max(dim=1).values)
            # print("  resolution:", resolution)

            # # fix layout
            # voxel_coords = coords.permute(0, 2, 1).contiguous()  # (B, N_occ, 3)
            # # clamp just in case
            # voxel_coords = voxel_coords.long().clamp(0, resolution - 1)

            outs = trilinear_devoxelize_cpu(
                grid=grid,
                coords=coords,
                resolution=resolution
            )  # (B, C, N)
            # build a flat‐index map for backward:
            idx_int = coords.long()  # assume coords in [0, R-1]
            idx_flat = (
                    idx_int[..., 0] * (resolution * resolution)
                    + idx_int[..., 1] * resolution
                    + idx_int[..., 2]
            )  # (B, N)
            # you can return dummy weights (they’re only used for grad–routing)
            B, C, N = outs.shape
            wgts = torch.ones(B, C, N, device=outs.device, dtype=outs.dtype)
            return outs, idx_flat, wgts

        @staticmethod
        def trilinear_devoxelize_backward(
                resolution: int,
                is_training: bool,
                coords: torch.Tensor,  # (B, N, 3)
                grid: torch.Tensor,  # placeholder, not used
                grad_out: torch.Tensor  # (B, C, R, R, R)
        ):
            # simply gather gradient at the nearest grid cell
            B, C, R1, R2, R3 = grad_out.shape
            R = R1
            idx_int = coords.long()
            idx_flat = (
                    idx_int[..., 0] * (R * R)
                    + idx_int[..., 1] * R
                    + idx_int[..., 2]
            )  # (B, N)
            grad_flat = grad_out.view(B, C, -1)  # (B, C, R^3)
            grad_feats = torch.gather(
                grad_flat, 2,
                idx_flat.unsqueeze(1).expand(-1, C, -1)
            )  # (B, C, N)
            return grad_feats

        @staticmethod
        def avg_voxelize_forward(features, coords, resolution):
            # match the real API
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

# NOTE: This is the custom CUDA C++ code for the sparse window attention mechanism
# import os
#
# from torch.utils.cpp_extension import load
#
# _src_path = os.path.dirname(os.path.abspath(__file__))
# _backend = load(name='_pvt_backend',
#                 extra_cflags=['-O3', '-std=c++17'],
#                 sources=[os.path.join(_src_path,'src', f) for f in [
#                     'interpolate/neighbor_interpolate.cpp',
#                     'interpolate/neighbor_interpolate.cu',
#                     'interpolate/trilinear_devox.cpp',
#                     'interpolate/trilinear_devox.cu',
#                     'sampling/sampling.cpp',
#                     'sampling/sampling.cu',
#                     'voxelization/vox.cpp',
#                     'voxelization/vox.cu',
#                     'bindings.cpp',
#                 ]]
#                 )
#
# __all__ = ['_backend']
