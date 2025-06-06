import torch

class VoxelGridCentersMixin:
    """
    Mixin providing method to compute voxel grid centers.
    """

    def generate_voxel_grid_centers(self, resolution):
        """
        Returns a tensor of shape (V, 3) with the 3D center coordinates
        of each voxel in a cubic grid of shape (R x R x R), normalized to [-1, 1]^3.

        Args:
            resolution (int): Number of voxels per axis (R)

        Returns:
            Tensor: (V, 3), where V = R^3
        """
        # Generate voxel indices in 3D grid: shape (R, R, R, 3)
        grid = torch.stack(torch.meshgrid(
            torch.arange(resolution),
            torch.arange(resolution),
            torch.arange(resolution),
            indexing='ij'  # Makes indexing (x, y, z) order
        ), dim=-1).float()  # shape: (R, R, R, 3)

        # Reshape to flat list of voxel indices: (V, 3) where V = R^3
        grid = grid.reshape(-1, 3)  # e.g., (27000, 3) for R=30

        # Compute voxel centers in normalized [-1, 1]^3 space
        grid = (grid + 0.5) / resolution  # Normalize to (0, 1)
        grid = grid * 2 - 1  # Map to (-1, 1)

        voxel_centers = grid.unsqueeze(0).expand(self.args.batch_size, -1, -1)  # shape: (B, R^3, 3)
        return voxel_centers.to(self.args.device)