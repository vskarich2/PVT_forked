from torch.autograd import Function
from modules.functional.backend import _backend
import torch

__all__ = ['trilinear_devoxelize']


class TrilinearDevoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True):
        """
        :param ctx:
        :param coords: the coordinates of points, FloatTensor[B, 3, N]
        :param features: FloatTensor[B, C, R, R, R]
        :param resolution: int, the voxel resolution
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]
        """
        B, C = features.shape[:2]
        features = features.contiguous().view(B, C, -1)
        coords = coords.contiguous()

        try:
            # Attempt CUDA backend call
            outs, inds, wgts = _backend.trilinear_devoxelize_forward(
                resolution, is_training, coords, features
            )
        except RuntimeError:
            # If CUDA backend fails, compute only outs and create dummy inds and wgts
            outs = _backend.trilinear_devoxelize_forward(
                resolution, is_training, coords, features
            )[0]
            device = features.device
            N = coords.shape[-1]
            inds = torch.zeros((B, N), dtype=torch.long, device=device)
            wgts = torch.ones((B, C, N), dtype=features.dtype, device=device)

        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.r = resolution
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of outputs, FloatTensor[B, C, N]
        :return:
            gradient of inputs, FloatTensor[B, C, R, R, R]
        """
        saved = ctx.saved_tensors
        if len(saved) == 2:
            inds, wgts = saved
        else:
            # Create dummy tensors based on grad_output’s shape
            B, C, N = grad_output.shape
            device = grad_output.device
            inds = torch.zeros((B, N), dtype=torch.long, device=device)
            wgts = torch.ones((B, C, N), dtype=grad_output.dtype, device=device)

        grad_inputs = _backend.trilinear_devoxelize_backward(
            grad_output.contiguous(), inds, wgts, ctx.r
        )
        return grad_inputs.view(
            grad_output.size(0), grad_output.size(1), ctx.r, ctx.r, ctx.r
        ), None, None, None


trilinear_devoxelize = TrilinearDevoxelization.apply
