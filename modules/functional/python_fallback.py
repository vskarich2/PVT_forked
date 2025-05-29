# python_fallback.py
import torch.nn.functional as F
import math
import torch
def trilinear_devoxelize_forward_cpu(features, coords, resolution):
    """
    CPU stub matching signature:
      inputs: features (B, C, N), coords (B, N, 3), resolution
      outputs: outs (B, C, R, R, R), inds (B, N), wgts (B, C, N)
    """
    # reuse your existing trilinear_devoxelize_cpu logic, but return also inds & weights
    outs = trilinear_devoxelize_cpu(
        grid=features.view(features.size(0), features.size(1), resolution, resolution, resolution),
        coords=coords,
        resolution=resolution,
        training=False
    )
    # python_fallback.trilinear_devoxelize_cpu returns (B, C, N)
    # we need outs, plus dummy inds & wgts:
    #   inds = flat_idx from avg_voxelize_forward (not used later)
    #   wgts = the trilinear weights per point-channel (we can return ones)
    B, C, N = features.size()
    device, dtype = features.device, features.dtype
    inds = torch.zeros(B, N, dtype=torch.long, device=device)
    wgts = torch.ones(B, C, N, dtype=dtype, device=device)
    return outs, inds, wgts


def trilinear_devoxelize_backward_cpu(grad_out, coords, resolution):
    """
    CPU stub matching signature:
      inputs: grad_out (B, C, R, R, R), coords (B, N, 3), resolution
      output: grad_features (B, C, N)
    """
    # Simplest: distribute gradient equally (or just gather)
    # Here we just gather the gradient from the nearest voxel:
    B, C, R1, R2, R3 = grad_out.size()
    R = R1
    _, N, _ = coords.size()
    # compute flat voxel indices
    idx = coords.long()
    flat_idx = idx[...,0]*R*R + idx[...,1]*R + idx[...,2]  # (B, N)
    grad_flat = grad_out.view(B, C, -1)
    # gather
    grad_feats = torch.gather(
        grad_flat, 2,
        flat_idx.unsqueeze(1).expand(-1, C, -1)
    )
    return grad_feats

import torch

def avg_voxelize_forward_cpu(
    features: torch.Tensor,
    coords: torch.Tensor,
    resolution: int
):
    """
    CPU stub for avg_voxelize_forward. This function converts a point cloud
    (represented by features and coordinates) into a regular 3D voxel grid
    by averaging the features of all points that fall into the same voxel.

    Args:
      features (torch.Tensor): A tensor of shape (B, C, N) representing
                               per-point feature vectors.
                               - B: Batch size.
                               - C: Number of feature channels per point (e.g., RGB, normal, semantic label).
                               - N: Total number of points in the point cloud for a given batch.

      coords (torch.Tensor):   A tensor of shape (B, 3, N) containing
                               the integer voxel indices (x, y, z) for each point.
                               These coordinates are expected to be within the range [0, resolution-1]
                               along each axis.
                               - The first dimension (index 0) of the second axis is the x-coordinate.
                               - The second dimension (index 1) of the second axis is the y-coordinate.
                               - The third dimension (index 2) of the second axis is the z-coordinate.

      resolution (int):        An integer R, representing the number of voxels along each
                               axis of the cubic output grid (i.e., R x R x R voxels).

    Returns:
      out (torch.Tensor):      A tensor of shape (B, C, R, R, R) containing the
                               averaged feature vectors for each voxel. Each voxel (x,y,z)
                               will hold a C-dimensional vector that is the average of
                               all point features that landed in that voxel.

      flat_idx (torch.Tensor): A tensor of shape (B, N) containing the flattened
                               1D voxel index for each point. This is useful for
                               the backward pass or for debugging.

      counts (torch.Tensor):   A tensor of shape (B, R**3) containing the number
                               of points that fell into each flattened voxel slot.
                               This is used for normalization (averaging) and can also
                               be useful for understanding voxel occupancy.
    """
    # Unpack dimensions from the input 'features' tensor.
    # B: Batch size
    # C: Number of channels/features per point
    # N: Number of points in the point cloud
    B, C, N = features.shape

    # Store the resolution (R) for convenience.
    R = resolution

    # Get the device (e.g., 'cpu' or 'cuda') and data type (e.g., torch.float32)
    # from the input features tensor. This ensures that all newly created tensors
    # are on the same device and have the same data type, preventing common errors.
    device = features.device
    dtype = features.dtype

    # --- step 1: Convert 3D voxel coordinates into a single flat index per point ---
    """
    The voxel grid is conceptually 3D (R x R x R), but for efficient processing
    with scatter operations, it's often flattened into a 1D array of size R^3.
    This step calculates the corresponding 1D index for each point's 3D voxel coordinate.
    """
    # Extract the x-coordinates for all points in all batches.
    # 'coords' is (B, 3, N), so coords[:, 0, :] takes all batches, the first channel (x),
    # and all points. .long() casts to integer type, as indices must be integers.
    x_idx = coords[:, 0, :].long()  # Shape: (B, N)

    # Extract the y-coordinates for all points.
    y_idx = coords[:, 1, :].long()  # Shape: (B, N)

    # Extract the z-coordinates for all points.
    z_idx = coords[:, 2, :].long()  # Shape: (B, N)

    # Compute the flat (1D) index for each point.
    # This formula maps a 3D (x, y, z) coordinate to a unique 1D index in a
    # flattened R x R x R grid. It's similar to how elements are stored in
    # row-major order in a 3D array: index = x * (R*R) + y * R + z.
    flat_idx = x_idx * (R * R) + y_idx * R + z_idx  # Shape: (B, N)

    # --- step 2: Allocate accumulators for feature sums and point counts per voxel ---
    """
    Before we can average, we need to:
    1. Sum up all feature vectors for points that fall into the same voxel.
    2. Count how many points fall into each voxel.
    These tensors will temporarily hold these sums and counts.
    """
    # Initialize 'out_flat' to store the sum of features for each voxel.
    # It has shape (B, C, R^3). For each batch 'b' and each flattened voxel index 'k',
    # out_flat[b, :, k] will accumulate the sum of C-dimensional feature vectors.
    out_flat = torch.zeros(B, C, R**3, device=device, dtype=dtype)

    # Initialize 'counts' to store the number of points in each voxel.
    # It has shape (B, R^3). For each batch 'b' and each flattened voxel index 'k',
    # counts[b, k] will store the total number of points that landed in that voxel.
    # Using torch.int64 for counts is appropriate as it stores discrete numbers of points.
    counts = torch.zeros(B, R**3, device=device, dtype=torch.int64)

    # --- step 3: Scatter-add features and count points, processing batch by batch ---
    """
    This loop iterates through each item in the batch. For each batch item,
    it distributes the point features into their respective voxel slots and
    tallies the number of points per voxel.
    """
    for b in range(B):
        # Get the flat indices for the current batch 'b'. Shape: (N,)
        fi = flat_idx[b]

        # Get the features for the current batch 'b'. Shape: (C, N)
        fv = features[b]

        # Expand the flat indices 'fi' to match the channel dimension 'C' of 'fv'.
        # 'fi' is (N,), 'fv' is (C, N). To use scatter_add_ along dimension 1 (points/voxels),
        # the index tensor must broadcast to match the dimensions of the source tensor 'fv'.
        # unsqueeze(0) makes it (1, N), then expand(C, -1) makes it (C, N).
        fi_exp = fi.unsqueeze(0).expand(C, -1)

        # Perform the scatter-add operation.
        # out_flat[b] is (C, R^3).
        # dim=1 means we are scattering along the second dimension (the flattened voxel dimension).
        # fi_exp (C, N) provides the target indices in dim 1 for each element in fv (C, N).
        # fv (C, N) provides the values to add.
        # Essentially, for each point 'n' in the current batch, its feature vector fv[:, n]
        # is added to the corresponding voxel slot out_flat[b, :, fi[n]].
        out_flat[b].scatter_add_(1, fi_exp, fv)

        # Count the number of points that fall into each unique flat voxel index.
        # torch.bincount(input, minlength) counts occurrences of each non-negative integer
        # in the 'input' tensor. 'minlength=R**3' ensures the output tensor has a size
        # equal to the total number of possible voxels, even if some are empty.
        # The result is stored in counts[b], which will be a 1D tensor of shape (R^3,).
        counts[b] = torch.bincount(fi, minlength=R**3)

    # --- step 4: Divide sums by counts to get the final averages ---
    """
    After summing all features for points within each voxel, this step performs
    the actual averaging by dividing the sum by the number of points that contributed
    to that sum. This ensures that each voxel's feature vector is an average,
    not a sum, making it more robust to varying point densities.
    """
    # Create a boolean mask indicating which voxel slots received at least one point.
    # This is important to avoid division by zero for empty voxels (where counts is 0).
    nonzero = counts > 0  # Shape: (B, R^3). True where counts > 0, False otherwise.

    # Convert the 'counts' tensor to the same floating-point data type as 'features'
    # before performing division. This prevents integer division issues and ensures
    # the output features maintain their floating-point precision.
    counts_f = counts.to(dtype)  # Shape: (B, R^3)

    # Iterate through each batch again to perform the division.
    for b in range(B):
        # Get the 1D boolean mask for the current batch 'b'. Shape: (R^3,)
        nz = nonzero[b]

        # Perform the division only for voxels that are non-empty (where nz is True).
        # out_flat[b, :, nz] selects all channels (:) for the non-empty voxels (nz).
        # counts_f[b, nz] provides the corresponding counts for these non-empty voxels.
        # The division is element-wise and effectively calculates:
        # averaged_feature_vector_for_voxel_k = sum_of_features_for_voxel_k / num_points_in_voxel_k
        out_flat[b, :, nz] /= counts_f[b, nz]

    # --- step 5: Reshape the flattened voxel features back into a 3D grid ---
    """
    The features were accumulated in a flattened (1D) representation of the voxel grid
    (R^3). This final step reshapes them back into the desired 3D (R x R x R) structure.
    """
    # Reshape 'out_flat' from (B, C, R^3) to (B, C, R, R, R).
    # The .view() method creates a new view of the tensor with the specified shape,
    # sharing the underlying data with the original tensor (if possible).
    out = out_flat.view(B, C, R, R, R)

    # Return the averaged voxel features, the flattened point indices, and the point counts per voxel.
    return out, flat_idx, counts


import torch

def avg_voxelize_backward_cpu(
    grad_out: torch.Tensor,
    flat_idx: torch.Tensor,
    counts: torch.Tensor
):
    """
    CPU stub for avg_voxelize_backward. This function performs the backward pass
    (gradient computation) for the `avg_voxelize_forward_cpu` operation.
    Its purpose is to compute the gradients with respect to the input `features`
    of the forward pass, given the gradients from the subsequent layer (`grad_out`).

    In the forward pass (`avg_voxelize_forward_cpu`), multiple point features
    are averaged to produce a single feature vector for a voxel.
    The backward pass must distribute the gradient received by a voxel back
    to all the original points that contributed to that voxel's average.
    Since the forward operation was an average (division by count `k`),
    the gradient for each contributing point is the voxel's gradient
    multiplied by `1/k`.

    Args:
      grad_out (torch.Tensor): A tensor of shape (B, C, R, R, R) representing
                               the gradients flowing back from the layer that
                               consumed the output of the forward voxelization.
                               This is the gradient with respect to the averaged
                               voxel features.
                               - B: Batch size.
                               - C: Number of feature channels.
                               - R: Resolution of the voxel grid along each axis.

      flat_idx (torch.Tensor): A tensor of shape (B, N) containing the flattened
                               1D voxel index for each point. This tensor is
                               returned by the `avg_voxelize_forward_cpu` function
                               and is crucial for mapping gradients from voxels back to points.

      counts (torch.Tensor):   A tensor of shape (B, R**3) containing the number
                               of points that fell into each flattened voxel slot.
                               This tensor is also returned by the forward pass
                               and is used here to correctly scale the gradients
                               (i.e., divide by the number of contributing points).

    Returns:
      grad_features (torch.Tensor): A tensor of shape (B, C, N) representing
                                    the gradients with respect to the original
                                    per-point feature vectors (`features`) from
                                    the forward pass. These gradients will then
                                    be propagated further back in the network.

      None, None:                 Placeholders for gradients with respect to
                                    `coords` and `resolution`. In this context,
                                    `coords` (voxel assignments) and `resolution`
                                    are typically not learnable parameters, so
                                    their gradients are not computed or are
                                    considered zero.
    """
    # Unpack dimensions from the input 'grad_out' tensor.
    # B: Batch size
    # C: Number of channels/features
    # R1, R2, R3: Dimensions of the 3D voxel grid (should be R x R x R)
    B, C, R1, R2, R3 = grad_out.shape

    # Assert that all dimensions of the output grid are equal, confirming it's a cubic grid.
    assert R1 == R2 == R3
    # Store the resolution (R) for convenience.
    R = R1

    # Get the number of points (N) from the 'flat_idx' tensor.
    # N: Number of points in the point cloud for a given batch.
    N = flat_idx.shape[1]

    # --- step 1: Flatten the incoming gradient tensor ---
    """
    The `grad_out` tensor is in a 3D (R x R x R) format, but the point indices
    (`flat_idx`) are 1D. To efficiently map gradients from voxels back to points,
    we flatten the `grad_out` tensor to match the 1D structure of `flat_idx`.
    """
    # Reshape 'grad_out' from (B, C, R, R, R) to (B, C, R^3).
    # This aligns the gradient tensor with the flattened voxel representation
    # used internally in the forward pass and for indexing with 'flat_idx'.
    grad_flat = grad_out.view(B, C, -1)

    # Initialize 'grad_features' as a tensor of zeros.
    # This tensor will accumulate the gradients for each original point.
    # Its shape (B, C, N) matches the shape of the 'features' input to the forward pass.
    # It's crucial to ensure it's on the same device and has the same data type
    # as the incoming gradient for correct operations.
    grad_features = torch.zeros(B, C, N, device=grad_out.device, dtype=grad_out.dtype)

    # --- step 2: Distribute gradients back to individual points, batch by batch ---
    """
    This loop iterates through each item in the batch. For each batch item,
    it calculates how much of the voxel's gradient should be passed back to
    each point that contributed to that voxel.
    """
    for b in range(B):
        # Get the flat indices for the current batch 'b'. Shape: (N,)
        # Each element fi[n] tells us which voxel point 'n' belonged to.
        fi = flat_idx[b]

        # Get the counts of points per voxel for the current batch 'b'. Shape: (R^3,)
        # Each element cnt[k] tells us how many points were averaged into voxel 'k'.
        cnt = counts[b]

        # Initialize an 'inverse' tensor. This will store 1/count for non-empty voxels.
        # It's initialized with zeros, and only updated where counts are positive.
        inv = torch.zeros_like(cnt, dtype=grad_out.dtype) # Ensure dtype matches for division

        # Create a boolean mask for non-zero counts.
        # This identifies which voxels actually received points in the forward pass,
        # preventing division by zero for empty voxels.
        nz = cnt > 0

        # Calculate the inverse of counts (1/k) only for non-empty voxels.
        # This is the scaling factor for the gradient propagation.
        # If a voxel's feature was an average of 'k' points, then the gradient
        # from that voxel is distributed equally (scaled by 1/k) to each of those 'k' points.
        inv[nz] = 1.0 / cnt[nz]

        # Expand the 'inv' tensor to broadcast across the channel dimension.
        # 'inv' is (R^3,). We need to multiply it with 'grad_flat[b]' which is (C, R^3).
        # Unsqueezing at dimension 0 makes it (1, R^3), allowing it to broadcast
        # across the 'C' channels during multiplication.
        inv_exp = inv.unsqueeze(0)  # Shape: (1, R^3)

        # Calculate the gradient for each point.
        # This is the core operation of the backward pass for averaging.
        # 1. `grad_flat[b].mul(inv_exp)`: Multiplies the gradient of each voxel
        #    (for all channels) by its corresponding inverse count (1/k).
        #    This results in a tensor of shape (C, R^3) where each voxel's
        #    gradient is now scaled by 1/k.
        # 2. `.gather(1, fi.unsqueeze(0).expand(C, -1))`: This operation "gathers"
        #    these scaled voxel gradients back to their original point locations.
        #    - `grad_flat[b].mul(inv_exp)` is the source tensor (C, R^3).
        #    - `dim=1` means we are gathering along the flattened voxel dimension.
        #    - `fi.unsqueeze(0).expand(C, -1)` provides the indices. For each point 'n',
        #      it uses `fi[n]` to pick the scaled gradient from the voxel it belonged to.
        #      Since `fi` is (N,), it's expanded to (C, N) to match the channels.
        # The result `grad_features[b]` is a tensor of shape (C, N), where each
        # `grad_features[b, :, n]` is the gradient for point 'n'.
        grad_features[b] = grad_flat[b].mul(inv_exp).gather(1, fi.unsqueeze(0).expand(C, -1))

    # --- step 3: Return computed gradients and placeholders ---
    # The function returns the gradients with respect to the input features.
    # It also returns None for 'coords' and 'resolution' as their gradients
    # are typically not required or are considered zero in this context.
    return grad_features, None, None

def trilinear_devoxelize_cpu(grid, coords, resolution: int, training: bool = False, **kw):
    """
    CPU fallback for trilinear_devoxelize. This function performs trilinear
    interpolation to sample features from a 3D voxel grid at specified
    floating-point coordinates. It's essentially the inverse of voxelization,
    converting a dense voxel representation back to point-wise features.

    Trilinear interpolation works by taking a point's continuous (x, y, z)
    coordinates within a voxel grid and calculating a weighted average of the
    features from the 8 surrounding voxels (the corners of the cube containing
    the point). The weights are inversely proportional to the distance from
    the point to each corner.

    Args:
        grid (torch.Tensor):    A tensor of shape (B, C, R, R, R) representing
                                the 3D voxel grid from which features will be sampled.
                                - B: Batch size.
                                - C: Number of feature channels per voxel.
                                - R: Resolution of the cubic voxel grid along each axis.

        coords (torch.Tensor):  A tensor of shape (B, 3, N) containing the
                                floating-point 3D coordinates (x, y, z) for N points
                                in each batch. These coordinates are expected to be
                                in the range [0, R-1], where R is the resolution.
                                - The second dimension (index 0) is x-coordinate.
                                - The second dimension (index 1) is y-coordinate.
                                - The second dimension (index 2) is z-coordinate.
                                - N: Total number of points to sample features for.

        resolution (int):       An integer R, representing the size of the cubic
                                voxel grid along each dimension.

        training (bool):        A boolean flag, ignored in this CPU stub. It's
                                included for API compatibility with potential
                                GPU implementations that might behave differently
                                during training (e.g., for gradient computation).

        **kw:                   Arbitrary keyword arguments, ignored in this stub
                                for API compatibility.

    Returns:
        pts_feat (torch.Tensor): A tensor of shape (B, C, N) containing the
                                 trilinearly interpolated feature vectors for
                                 each of the N input coordinates. Each point 'n'
                                 will have a C-dimensional feature vector.
    """
    # --- Step 1: Pre-process input tensors and extract dimensions ---

    # Permute the dimensions of 'coords' from (B, 3, N) to (B, N, 3).
    # This makes it easier to unpack the x, y, z coordinates directly from the
    # last dimension (e.g., coords[..., 0] for x, coords[..., 1] for y, etc.).
    coords = coords.permute(0, 2, 1) # Shape: (B, N, 3)

    # Extract dimensions from the 'grid' tensor.
    # B: Batch size
    # C: Number of channels/features per voxel
    # R: Resolution (assuming R1=R2=R3 from grid.shape[2:])
    # Note: grid.shape will be (B, C, R, R, R), so grid.shape[0] is B, grid.shape[1] is C,
    # and grid.shape[2] is R. The remaining R's are implicitly handled by the view.
    B, C, R_dummy = grid.shape[0], grid.shape[1], grid.shape[2] # Extract B, C from grid
    # Verify the resolution matches the input parameter.
    assert R_dummy == resolution, "Grid resolution mismatch with input 'resolution' parameter."
    R = resolution # Use the provided resolution for consistency.

    # Extract the number of points (N) from the permuted 'coords' tensor.
    # The first underscore is for batch size (already B), the second for N, third for 3.
    _, N, _ = coords.shape # Shape: (B, N, 3)

    # Flatten the spatial dimensions of the 'grid' tensor for easier feature gathering.
    # The original grid is (B, C, R, R, R). By viewing it as (B, C, R^3), we can
    # use a single 1D index to access any voxel's features across all channels.
    grid_flat = grid.view(B, C, -1)  # Shape: (B, C, R^3)

    # --- Step 2: Clamp coordinates and determine surrounding voxel indices ---

    # Clamp the input coordinates to ensure they are within the valid range [0, R-1].
    # This prevents out-of-bounds indexing errors if coordinates are slightly outside
    # the grid boundaries due to floating-point inaccuracies or other reasons.
    coords_clamped = coords.clamp(0, R - 1) # Shape: (B, N, 3)

    # Split the clamped coordinates into separate x, y, and z components.
    # .unbind(-1) splits the tensor along its last dimension (dimension 2, which has size 3).
    # Each resulting tensor (x, y, z) will have shape (B, N).
    x, y, z = coords_clamped.unbind(-1)  # Each shape: (B, N)

    # Calculate the floor and ceiling integer indices for each coordinate.
    # These represent the 8 corners of the voxel cube that encloses each point.
    # x0, y0, z0 are the "lower" integer coordinates (floor).
    x0 = x.floor().long() # Shape: (B, N)
    y0 = y.floor().long() # Shape: (B, N)
    z0 = z.floor().long() # Shape: (B, N)

    # x1, y1, z1 are the "upper" integer coordinates (ceil).
    # We clamp them to R-1 to ensure they don't exceed the grid boundaries,
    # especially for points exactly on the upper boundary (e.g., x=R-1).
    x1 = (x0 + 1).clamp(max=R - 1) # Shape: (B, N)
    y1 = (y0 + 1).clamp(max=R - 1) # Shape: (B, N)
    z1 = (z0 + 1).clamp(max=R - 1) # Shape: (B, N)

    # --- Step 3: Compute fractional weights for interpolation ---

    # Calculate the fractional part of each coordinate.
    # These values (xd, yd, zd) represent how far the point is from the
    # '0' (floor) corner along each axis, ranging from [0.0, 1.0).
    # For trilinear interpolation, these fractions directly serve as weights.
    # .unsqueeze(1) adds a singleton dimension, making them (B, 1, N). This
    # allows for broadcasting during the multiplication with feature vectors (B, C, N).
    xd = (x - x0.float()).unsqueeze(1)  # Shape: (B, 1, N)
    yd = (y - y0.float()).unsqueeze(1)  # Shape: (B, 1, N)
    zd = (z - z0.float()).unsqueeze(1)  # Shape: (B, 1, N)

    # --- Step 4: Define a helper for 1D flat index calculation ---

    # Helper function to convert 3D integer voxel coordinates (xi, yi, zi)
    # into a single 1D flat index within the R x R x R grid.
    # This formula is standard for mapping 3D indices to a 1D array in row-major order.
    def _flat_idx(xi, yi, zi):
        # xi, yi, zi are tensors of shape (B, N)
        return xi * (R * R) + yi * R + zi  # Result shape: (B, N)

    # --- Step 5: Calculate flat indices for all 8 corners of the bounding voxel ---

    # Each point (x,y,z) is enclosed by a cube defined by 8 corners.
    # We calculate the 1D flat index for each of these 8 corners for every point.
    # The naming convention (e.g., idx000) corresponds to (x0, y0, z0), (x0, y0, z1), etc.
    idx000 = _flat_idx(x0, y0, z0) # Shape: (B, N)
    idx001 = _flat_idx(x0, y0, z1) # Shape: (B, N)
    idx010 = _flat_idx(x0, y1, z0) # Shape: (B, N)
    idx011 = _flat_idx(x0, y1, z1) # Shape: (B, N)
    idx100 = _flat_idx(x1, y0, z0) # Shape: (B, N)
    idx101 = _flat_idx(x1, y0, z1) # Shape: (B, N)
    idx110 = _flat_idx(x1, y1, z0) # Shape: (B, N)
    idx111 = _flat_idx(x1, y1, z1) # Shape: (B, N)

    # --- Step 6: Gather features from the 8 corners of the grid ---

    # Helper function to gather feature vectors from the flattened grid using 1D indices.
    def _gather(idx):
        # 'idx' is a tensor of shape (B, N) containing the 1D flat indices for each point.
        # To use torch.gather, the index tensor needs to have the same number of dimensions
        # as the source tensor ('grid_flat') and broadcast appropriately.
        # 'grid_flat' is (B, C, R^3). We want to gather along dimension 2 (R^3).
        # So, we unsqueeze 'idx' to (B, 1, N) and then expand it to (B, C, N).
        # This makes sure that for each point, we gather its C-dimensional feature vector.
        return torch.gather(
            grid_flat, # Source tensor: (B, C, R^3)
            2,         # Dimension to gather along (the flattened spatial dimension)
            idx.unsqueeze(1).expand(-1, C, -1) # Index tensor: (B, C, N)
        )
        # Resulting shape: (B, C, N) for each gathered corner's features.

    # Gather the C-dimensional feature vectors for each of the 8 corners for all points.
    v000 = _gather(idx000) # Features from (x0, y0, z0) corner. Shape: (B, C, N)
    v001 = _gather(idx001) # Features from (x0, y0, z1) corner. Shape: (B, C, N)
    v010 = _gather(idx010) # Features from (x0, y1, z0) corner. Shape: (B, C, N)
    v011 = _gather(idx011) # Features from (x0, y1, z1) corner. Shape: (B, C, N)
    v100 = _gather(idx100) # Features from (x1, y0, z0) corner. Shape: (B, C, N)
    v101 = _gather(idx101) # Features from (x1, y0, z1) corner. Shape: (B, C, N)
    v110 = _gather(idx110) # Features from (x1, y1, z0) corner. Shape: (B, C, N)
    v111 = _gather(idx111) # Features from (x1, y1, z1) corner. Shape: (B, C, N)

    # --- Step 7: Compute the trilinear interpolation weights ---

    # These weights are derived from the fractional parts (xd, yd, zd).
    # For a point (x,y,z) within a unit cube defined by (x0,y0,z0) to (x1,y1,z1),
    # the weights are the products of (1-fractional_dist) or (fractional_dist)
    # along each axis.
    # For example, w000 = (1-xd)*(1-yd)*(1-zd) means the weight for the (x0,y0,z0) corner.
    # Each weight tensor will have shape (B, 1, N), allowing it to broadcast
    # with the (B, C, N) feature vectors during multiplication.
    w000 = (1 - xd) * (1 - yd) * (1 - zd) # Weight for (x0, y0, z0)
    w001 = (1 - xd) * (1 - yd) * zd      # Weight for (x0, y0, z1)
    w010 = (1 - xd) * yd * (1 - zd)      # Weight for (x0, y1, z0)
    w011 = (1 - xd) * yd * zd            # Weight for (x0, y1, z1)
    w100 = xd * (1 - yd) * (1 - zd)      # Weight for (x1, y0, z0)
    w101 = xd * (1 - yd) * zd            # Weight for (x1, y0, z1)
    w110 = xd * yd * (1 - zd)            # Weight for (x1, y1, z0)
    w111 = xd * yd * zd                  # Weight for (x1, y1, z1)

    # --- Step 8: Perform the weighted sum of corner features ---

    # The final interpolated feature for each point is the sum of the
    # 8 gathered corner features, each multiplied by its corresponding trilinear weight.
    # This is the core of the trilinear interpolation.
    pts_feat = (
            v000 * w000 + # (B, C, N) * (B, 1, N) = (B, C, N) due to broadcasting
            v001 * w001 +
            v010 * w010 +
            v011 * w011 +
            v100 * w100 +
            v101 * w101 +
            v110 * w110 +
            v111 * w111
    )  # Final shape: (B, C, N)

    # Return the trilinearly interpolated features for each point.
    return pts_feat


def sparse_window_attention_cpu(
    voxel_feats: torch.Tensor,
    window_size: int,
    num_heads: int,
    head_dim: int,
    **kwargs
) -> torch.Tensor:
    """
    Pure‐Python fallback for PVT’s sparse_window_attention.

    Args:
      voxel_feats: (B, N, C) where C = num_heads * head_dim
      window_size: ignored in this stub
      num_heads: number of attention heads
      head_dim:   dimension per head
      **kwargs:   any other args from the real stub

    Returns:
      Tensor of shape (B, N, C): the “attended” features (here full attention)
    """
    B, N, C = voxel_feats.shape
    assert C == num_heads * head_dim, "C must equal num_heads*head_dim"

    # 1) split into heads
    x = voxel_feats.view(B, N, num_heads, head_dim)            # (B, N, H, d_h)
    x = x.permute(0, 2, 1, 3)                                  # (B, H, N, d_h)

    # 2) compute Q, K, V (here identity projections)
    Q = x  # (B, H, N, d_h)
    K = x  # (B, H, N, d_h)
    V = x  # (B, H, N, d_h)

    # 3) dot‐product attention scores
    #    (B, H, N, d_h) @ (B, H, d_h, N) -> (B, H, N, N)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(head_dim)
    attn   = F.softmax(scores, dim=-1)                         # (B, H, N, N)

    # 4) weighted sum
    out = torch.matmul(attn, V)                                # (B, H, N, d_h)

    # 5) merge heads back
    out = out.permute(0, 2, 1, 3).contiguous()                 # (B, N, H, d_h)
    out = out.view(B, N, C)                                    # (B, N, C)

    return out

