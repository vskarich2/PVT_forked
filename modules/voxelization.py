import torch
import torch.nn as nn
import modules.functional as F  # Assumes custom voxel ops like avg_voxelize

__all__ = ['Voxelization']


class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)      # Voxel grid resolution (R) along each axis → total voxels = R³
        self.normalize = normalize    # Whether to normalize input coordinates to unit cube
        self.eps = eps                # Small epsilon to avoid division by zero in normalization

    def forward(self, features, coords):
        """
        features: (B, C, N) – input point features (e.g., xyz + normals)
        coords: (B, 3, N) – input point coordinates in 3D space
        """

        coords = coords.detach()  # Prevent gradients from flowing through coordinates

        # Center the point cloud by subtracting the per-sample mean across N points
        norm_coords = coords - coords.mean(2, keepdim=True)  # shape: (B, 3, N)

        if self.normalize:
            # Normalize coordinates to [-0.5, 0.5] and then shift to [0, 1]
            # First normalize each sample by the *largest* norm of any point (for uniform scaling)
            norm_factor = norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values
            norm_coords = norm_coords / (norm_factor * 2.0 + self.eps) + 0.5
        else:
            # Assume input already roughly in [-1, 1] range; shift to [0, 1]
            norm_coords = (norm_coords + 1) / 2.0

        # Scale to voxel grid: multiply by R and clamp to valid voxel indices
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)

        # vox_coords: shape (B, 3, N) gives voxel indices per point, NOT per voxel.
        # We round to nearest integer voxel indices, mapping point coordinates to voxel
        # indices.
        vox_coords = torch.round(norm_coords).to(torch.int32)  # shape: (B, 3, N)

        # Call a custom voxel aggregation function: average point features within each voxel
        # This returns a dense voxel grid of shape (B, C, R, R, R)
        avg_voxel_features = F.avg_voxelize(features, vox_coords, self.r)
        return avg_voxel_features, norm_coords

    def extra_repr(self):
        # Custom string representation for printing
        return 'resolution={}{}'.format(
            self.r,
            ', normalized eps = {}'.format(self.eps) if self.normalize else ''
        )
