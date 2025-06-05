# modules/functional/devoxelization.py

from torch.autograd import Function
from modules.functional.backend import _backend
import torch

__all__ = ['trilinear_devoxelize']


class TrilinearDevoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True):
        """
        :param ctx:
        :param coords:     FloatTensor[B, 3, N]  (point coordinates)
        :param features:   FloatTensor[B, C, R, R, R]  (voxel grid)
        :param resolution: int, the voxel resolution R
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]   (point‐wise features after trilinear interpolation)
        """
        B, C = features.shape[:2]

        # We pass the voxel grid as‐is (shape: B × C × R × R × R).
        # The backend implementation will return (outs, inds, wgts).
        outs, inds, wgts = _backend.trilinear_devoxelize_forward(
            resolution,       # int
            is_training,      # bool
            coords.contiguous(),   # (B, 3, N)
            features.contiguous()   # (B, C, R, R, R)
        )

        # Always save these two tensors for backward:
        #  - inds: flat‐index per point (B, N)
        #  - wgts: interpolation weights (B, C, N)
        ctx.save_for_backward(inds, wgts)
        ctx.r = resolution
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: FloatTensor[B, C, N]
        :return:
            grad_features: FloatTensor[B, C, R, R, R]  (gradient w.r.t. the voxel grid)
            None, None, None  (no gradients for coords, resolution, is_training)
        """
        # Unpack the two saved tensors:
        inds, wgts = ctx.saved_tensors  # both exist now, guaranteed

        # Call the backend backward function:
        grad_inputs = _backend.trilinear_devoxelize_backward(
            grad_output.contiguous(),  # (B, C, N)
            inds,                      # (B, N)
            wgts,                      # (B, C, N)
            ctx.r                      # resolution R
        )
        # grad_inputs comes back as (B, C, R^3).  Reshape to (B, C, R, R, R):
        return grad_inputs.view(
            grad_output.size(0),    # B
            grad_output.size(1),    # C
            ctx.r,                  # R
            ctx.r,                  # R
            ctx.r                   # R
        ), None, None, None


trilinear_devoxelize = TrilinearDevoxelization.apply
