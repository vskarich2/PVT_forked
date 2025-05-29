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
        coords = coords.detach()
        print(f"[Voxelization] input coords shape: {coords.shape}")
        print(f"  raw coords    min/max: {coords.min().item():.6f} / {coords.max().item():.6f}")

        norm_coords = coords - coords.mean(2, keepdim=True)
        print(
            f"[Voxelization] centered coords    min/max: {norm_coords.min().item():.6f} / {norm_coords.max().item():.6f}")

        if self.normalize:
            norms = norm_coords.norm(dim=1, keepdim=True)
            max_norm = norms.max(dim=2, keepdim=True).values
            denom = max_norm * 2.0 + self.eps
            norm_coords = norm_coords / denom + 0.5
            print(
                f"[Voxelization] normalized coords min/max: {norm_coords.min().item():.6f} / {norm_coords.max().item():.6f}")
            print(f"  denom stats   min/max: {denom.min().item():.6f} / {denom.max().item():.6f}")
        else:
            norm_coords = (norm_coords + 1) / 2.0
            print(
                f"[Voxelization] simple-norm coords min/max: {norm_coords.min().item():.6f} / {norm_coords.max().item():.6f}")

        norm_coords = norm_coords * self.r
        print(
            f"[Voxelization] after scaling       min/max: {norm_coords.min().item():.6f} / {norm_coords.max().item():.6f}")
        norm_coords = torch.clamp(norm_coords, 0, self.r - 1)
        print(
            f"[Voxelization] after clamping     min/max: {norm_coords.min().item():.6f} / {norm_coords.max().item():.6f}")

        vox_coords = torch.round(norm_coords).long()
        print(f"[Voxelization] vox_coords shape: {vox_coords.shape}")
        sample = vox_coords[0, :, :min(5, vox_coords.shape[-1])].cpu().numpy().T
        print(f"  sample voxel indices (first 5):\n{sample}")

        coords_Nx3 = vox_coords[0].permute(1, 0)
        unique_bins = torch.unique(coords_Nx3, dim=0)
        print(f"[Voxelization] # unique voxel bins before avg: {unique_bins.shape[0]}")

        # average into voxels
        averaged_voxel_features = F.avg_voxelize(features, vox_coords, self.r)
        print(f"[Voxelization] output features shape: {averaged_voxel_features.shape}")

        # count non-empty voxels after averaging
        # assume averaged_voxel_features is [B, C, ...voxel_dims...]
        B, C = averaged_voxel_features.shape[:2]
        # flatten all spatial dims into one
        flat = averaged_voxel_features.view(B, C, -1)
        non_empty_mask = (flat.abs().sum(dim=1) > 0)  # shape (B, num_voxels)
        non_empty_counts = non_empty_mask.sum(dim=1)  # one count per batch
        print(f"[Voxelization] # non-empty voxels after avg: {non_empty_counts.tolist()}")

        return averaged_voxel_features, norm_coords

    # def forward(self, features, coords):
    #     coords = coords.detach()
    #     norm_coords = coords - coords.mean(2, keepdim=True)
    #     if self.normalize:
    #         norm_coords = norm_coords / (norm_coords.norm(dim=1, keepdim=True).max(dim=2, keepdim=True).values * 2.0 + self.eps) + 0.5
    #     else:
    #         norm_coords = (norm_coords + 1) / 2.0
    #     norm_coords = torch.clamp(norm_coords * self.r, 0, self.r - 1)
    #     vox_coords = torch.round(norm_coords).to(torch.int32)
    #     # vox_coords is shape (1,3,N), so reshape to N×3
    #
    #     averaged_voxel_features = F.avg_voxelize(features, vox_coords, self.r)
    #
    #     return averaged_voxel_features, norm_coords

    def extra_repr(self):
        return 'resolution={}{}'.format(self.r, ', normalized eps = {}'.format(self.eps) if self.normalize else '')
