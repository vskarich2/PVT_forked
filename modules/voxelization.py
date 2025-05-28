import torch
import torch.nn as nn
import modules.functional as F

__all__ = ['Voxelization']


class Voxelization(nn.Module):
    def __init__(self, resolution, normalize=True, eps=0):
        super().__init__()
        self.r = int(resolution)
        self.normalize = normalize
        self.eps = eps

    def forward(self, features, coords):
        # features: (B, C_pt, N)  per-point feature tensor
        # coords:   (B, 3, N)     per-point 3D coordinates

        # Detach coords from the autograd graph so they aren't tracked during voxelization
        coords = coords.detach()

        # Center the point cloud by subtracting the per-point-set mean over N
        # dims: coords.mean(2, keepdim=True) → (B, 3, 1), broadcasting back to (B, 3, N)
        norm_coords = coords - coords.mean(2, keepdim=True)

        if self.normalize:
            # Compute the maximum L2 norm across all points & batches for normalization
            # norm_coords.norm(dim=1, keepdim=True) → (B, 1, N) gives per-point distance from origin
            # .max(dim=2, keepdim=True).values → (B, 1, 1) is the largest radius in each batch
            max_radius = norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values

            # Normalize to unit sphere, then shift into [0,1] range
            # Dividing by (2*max_radius + eps) scales coords to roughly [–0.5,0.5], +0.5 → [0,1]
            norm_coords = norm_coords / (max_radius * 2.0 + self.eps) + 0.5
        else:
            # If not normalizing by max radius, simply shift centered coords from [–1,1] to [0,1]
            norm_coords = (norm_coords + 1.0) / 2.0

        # Scale normalized coordinates into [0, R–1] for a grid of resolution R
        norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)

        # Round to nearest integer voxel indices and cast to int
        # vox_coords: (B, N, 3) with values in 0..R-1
        vox_coords = torch.round(norm_coords).to(torch.int32)

        # Perform average pooling of point-features into the R×R×R voxel grid
        # Returns:
        #   - a dense tensor (B, C_pt, R, R, R) of averaged features per voxel
        #   - the floating-point norm_coords (B, 3, N) for later devoxelization
        return F.avg_voxelize(features, vox_coords, self.r), norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
