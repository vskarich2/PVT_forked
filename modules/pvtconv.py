# Assuming these are custom modules defined elsewhere, providing their functionality.
# F: Likely a functional module for operations like trilinear_devoxelize.
# Voxelization: Module to convert point clouds to voxel grids.
# SharedTransformer: A transformer module for point features.
# SE3d: A 3D Squeeze-and-Excitation module for channel-wise re-weighting.
import torch.nn as nn

import modules.functional as F
from modules.se import SE3d
from modules.shared_transformer import SharedTransformer
from modules.voxelization import Voxelization

# Defines the public API of this module, specifying which classes are exposed
# when `from . import module_name` is used.
__all__ = ['PVTConv','PartPVTConv','SemPVTConv']

from modules.voxel_encoder import VoxelEncoder, SegVoxelEncoder

class PVTConv(nn.Module):
    """
    PVTConv (Point-Voxel Transformer Convolution) is a hybrid architecture
    that processes 3D point cloud data by combining two parallel pathways:
    1. A **Voxel Path**: Converts points to a voxel grid, processes it with
       3D convolutions and windowed self-attention, and then devoxelizes back to points.
    2. A **Point Path**: Processes raw point features using a SharedTransformer
       (likely a point-based self-attention or MLP).

    The outputs of these two paths are then fused. This design aims to leverage
    the strengths of both grid-based (local context, convolution efficiency)
    and point-based (fine-grained detail, permutation invariance) processing.
    """
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            kernel_size,
            resolution,
            normalize=True,
            eps=0
    ):
        """
        Initializes the PVTConv module.

        Args:
            in_channels (int): Number of input feature channels for points.
            out_channels (int): Number of output feature channels for points.
            kernel_size (int): Kernel size for initial voxel convolution.
            resolution (int): Resolution of the 3D voxel grid.
            normalize (bool): Whether to normalize coordinates in Voxelization.
            eps (float): Epsilon for normalization in Voxelization.
        """
        super().__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.boxsize = 3 # Default box size for windowed attention
        self.mlp_dims = out_channels # Default MLP hidden dimension
        self.drop_path1 = 0.1 # DropPath rate for Transformer
        self.drop_path2 = 0.2 # DropPath rate for Transformer

        # Voxelization module: Converts point features and coordinates into a dense voxel grid.
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)

        # Voxel Encoder: Processes the voxel grid using 3D convolutions and Transformer blocks.
        self.voxel_encoder = VoxelEncoder(
            in_channels,
            out_channels,
            kernel_size,
            resolution,
            self.boxsize,
            self.mlp_dims,
            self.drop_path1,
            self.drop_path2,
            self.args
        )

        # SE3d (Squeeze-and-Excitation) module: Applies channel-wise recalibration
        # to the voxel features, enhancing important channels and suppressing less relevant ones.
        self.SE = SE3d(out_channels)

        # SharedTransformer for point-based features: Processes the original point features.
        self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        """
        Performs the forward pass of the PVTConv module.

        Args:
            inputs (tuple): A tuple containing:
                            - features (torch.Tensor): (B, C_in, N) point-wise features.
                            - coords (torch.Tensor): (B, 3, N) original point coordinates (float).

        Returns:
            tuple: A tuple containing:
                   - fused_features (torch.Tensor): (B, C_out, N) fused per-point features.
                   - coords (torch.Tensor): (B, 3, N) original point coordinates (unchanged).
        """
        # Unpack the input tuple into point features and coordinates.
        features, coords = inputs # features: (B, C_in, N), coords: (B, 3, N)
        if self.args.scanobject_compare:
            print("PVTConv 1")
        # -------------------------------
        # 1) Voxel path: Convert points → voxels → process → devoxelize back to points
        # -------------------------------

        # a) Voxelization: Bin points into a dense R³ grid.
        #    This operation aggregates point features into voxel cells (e.g., by averaging).
        #    - voxel_features: (B, C_voxel, R, R, R) - the 3D voxel grid features.
        #    - voxel_coords:   (B, 3, N) - integer voxel indices for each original point.
        #                      These are crucial for the devoxelization step.
        if self.args.scanobject_compare:
            print("PVTConv 2")
        voxel_features, voxel_coords = self.voxelization(features, coords)
        # b) Voxel encoder: Apply 3D convolutions and local-window self-attention.
        #    This module processes the dense voxel grid, extracts hierarchical features,
        #    and aggregates local context within the grid.
        #    The output remains a 3D voxel grid of shape (B, C_voxel, R, R, R).
        if self.args.scanobject_compare:
            print("PVTConv 3")
        voxel_features = self.voxel_encoder(voxel_features)

        # c) Squeeze-and-Excitation (SE): Channel-wise gating.
        #    This module adaptively re-weights each channel of the voxel features,
        #    selectively emphasizing informative channels.
        if self.args.scanobject_compare:
            print("PVTConv 4")
        voxel_features = self.SE(voxel_features)

        # d) Trilinear devoxelization: Interpolate voxel grid features back to the original N points.
        #    This uses the `voxel_coords` (integer indices of the voxels each point belongs to)
        #    to perform trilinear interpolation, effectively sampling features from the continuous
        #    voxel grid at the exact locations of the original points.
        #    The output is per-point features of shape (B, C_voxel, N).
        if self.args.scanobject_compare:
            print("PVTConv 5")
        voxel_features = F.trilinear_devoxelize(
            voxel_features,
            voxel_coords,
            self.resolution,
            self.training,
            self.args.scanobject_compare
        )

        # -------------------------------
        # 2) Point path: Pure point-cloud self-attention
        # -------------------------------
        if self.args.scanobject_compare:
            print("PVTConv 6")
        # a) Rearrange coordinates to (B, N, 3) for pairwise difference computation.
        #    The original 'coords' are (B, 3, N).
        pos = coords.permute(0, 2, 1) # Shape: (B, N, 3)

        # b) Compute pairwise relative positions.
        #    This calculates the vector difference between all pairs of points within a batch.
        #    - pos[:, :, None, :] is (B, N, 1, 3)
        #    - pos[:, None, :, :] is (B, 1, N, 3)
        #    Subtraction results in (B, N, N, 3), where (i, j) element is (pos[i] - pos[j]).
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        # Sum over the last dimension (3, for x,y,z components) to get a scalar bias for each pair.
        # This yields a (B, N, N) matrix, where each element (i, j) is a scalar bias
        # representing the "distance" or "relation" between point i and point j.
        rel_pos = rel_pos.sum(dim=-1) # Shape: (B, N, N)

        # Apply the SharedTransformer to the original point features, incorporating relative positions.
        # This path directly processes the point cloud, often using self-attention
        # that can leverage the relative positional information.
        # The output 'point_out' has shape (B, C_out, N).
        point_out = self.point_features(features, rel_pos)

        # -------------------------------
        # 3) Fuse voxel + point outputs
        # -------------------------------

        # Element-wise sum of the features from the two parallel paths.
        # This combines the spatially contextualized features from the voxel path
        # with the fine-grained, permutation-invariant features from the point path.
        # Both 'voxel_features' and 'point_out' are (B, C_out, N).
        if self.args.no_point_attention:
            fused_features = voxel_features
        else:
            fused_features = voxel_features + point_out
        # Return the fused per-point features and the original (unchanged) coordinates.
        return fused_features, coords

class PartPVTConv(nn.Module):
    """
    A variant of PVTConv, possibly tailored for part segmentation or other tasks
    where the point path might be simpler (e.g., not using relative positions).
    The main difference from PVTConv is in the `point_features` module's forward call.
    """
    def __init__(self, in_channels, out_channels, kernel_size, resolution, normalize=True, eps=0):
        """
        Initializes PartPVTConv. Parameters are similar to PVTConv.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.boxsize = 3
        self.mlp_dims = out_channels
        self.drop_path1 = 0.1
        self.drop_path2 = 0.1 # Slightly different drop_path2 from PVTConv

        # Voxelization, VoxelEncoder, SE3d are the same as in PVTConv.
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        self.voxel_encoder = VoxelEncoder(in_channels, out_channels, kernel_size, resolution, self.boxsize,
                                             self.mlp_dims, self.drop_path1, self.drop_path2)
        self.SE = SE3d(out_channels)
        # SharedTransformer for point-based features.
        self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        """
        Performs the forward pass of PartPVTConv.

        Args:
            inputs (tuple): A tuple containing:
                            - features (torch.Tensor): (B, C_in, N) point-wise features.
                            - coords (torch.Tensor): (B, 3, N) original point coordinates.

        Returns:
            tuple: A tuple containing:
                   - fused_features (torch.Tensor): (B, C_out, N) fused per-point features.
                   - coords (torch.Tensor): (B, 3, N) original point coordinates (unchanged).
        """
        features, coords = inputs
        # Voxel path is identical to PVTConv.
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_encoder(voxel_features)
        voxel_features = self.SE(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)

        # Fusion: The key difference is here.
        # The `SharedTransformer` in this variant is called with only `features`,
        # implying it might not use relative positional information or has a
        # simpler attention mechanism compared to PVTConv's point path.
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords

class SemPVTConv(nn.Module):
    """
    A variant of PVTConv, likely designed for semantic segmentation tasks.
    It incorporates the `SegVoxelEncoder`, which includes a CutMix-like
    data augmentation strategy.
    """
    def __init__(self, in_channels, out_channels, kernel_size, resolution, normalize=True, eps=0):
        """
        Initializes SemPVTConv. Parameters are similar to PVTConv.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.boxsize = 4 # Different box size (4)
        self.mlp_dims = out_channels * 4 # Larger MLP hidden dimension
        self.drop_path1 = 0. # Different DropPath rates
        self.drop_path2 = 0.1

        # Voxelization module.
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)

        # Voxel Encoder: Uses SegVoxelEncoder, which includes CutMix augmentation.
        self.voxel_encoder = SegVoxelEncoder(in_channels, out_channels, kernel_size, resolution, self.boxsize,
                                             self.mlp_dims, self.drop_path1, self.drop_path2)

        # SE3d module.
        self.SE = SE3d(out_channels)
        # SharedTransformer for point-based features (same as PartPVTConv, no rel_pos).
        self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        """
        Performs the forward pass of SemPVTConv.

        Args:
            inputs (tuple): A tuple containing:
                            - features (torch.Tensor): (B, C_in, N) point-wise features.
                            - coords (torch.Tensor): (B, 3, N) original point coordinates.

        Returns:
            tuple: A tuple containing:
                   - fused_features (torch.Tensor): (B, C_out, N) fused per-point features.
                   - coords (torch.Tensor): (B, 3, N) original point coordinates (unchanged).
        """
        features, coords = inputs
        # Voxel path: Uses SegVoxelEncoder which applies CutMix.
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_encoder(voxel_features) # CutMix happens here
        voxel_features = self.SE(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)

        # Point path and fusion: Identical to PartPVTConv.
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords
