# dynamic_sparse_voxel_test.py
"""
Test script to verify dynamic-sparse kNN on voxelized point clouds
only on non-empty voxels.

Workflow:
1) Load a single ModelNet40 .txt file (N × 6: xyz + normals).
2) Voxelize points into a dense R³ grid via the Voxelization module.
3) Flatten the voxel grid into V = R³ tokens with D_in features per token.
4) Compute each voxel’s continuous 3D center in [-1,1]³.
5) Filter out empty voxels so we only process true data-bearing cells.
6) Run dynamic-sparse kNN on the non-empty voxel tokens to get neighbors.
7) Build edge features and score each edge via a small MLP.
8) Print tensor shapes and sample neighbor indices for verification.
"""
import os
import numpy as np
import torch

from PVT_forked_repo.PVT_forked.constants import BASE_DIR, DATA_DIR, NUM_POINTS_TEST
from PVT_forked_repo.PVT_forked.data import pc_normalize
from modules.voxelization import Voxelization           # Voxelization module
from knn_scoring import knn_filter_empty, build_edge_features, EdgeScorer


def main(txt_path: str, resolution: int = 30, k: int = 100, eps=1e-6):
    """
    Processes a point cloud file, voxelizes it, finds k-nearest neighbors for non-empty
    voxels, and computes edge scores between them.

    Args:
        txt_path:    Path to a ModelNet40 .txt file with N rows of xyz,nx,ny,nz.
        resolution:  Number of bins per axis for the voxel grid (R).
        k:           Number of nearest neighbors per voxel token.
    """
    # --- 1) Load the point-cloud + normals ---
    # Purpose: To read the raw 3D point data (XYZ coordinates) and associated
    # surface normal vectors (nx, ny, nz) from the specified text file.
    # Surface normals provide information about the orientation of the surface at each point,
    # which can be a valuable feature for shape understanding.
    data = np.loadtxt(txt_path, delimiter=',', dtype=np.float32)  # Shape: (NumTotalPoints, 6)

    # Purpose: To ensure that the subsampling in the next step is random and not biased
    # by any ordering present in the original file. This helps in obtaining a more
    # representative sample of the point cloud.
    np.random.shuffle(data)

    # Purpose: To subsample the point cloud to a fixed number of points (NUM_POINTS_TEST, e.g., 1024).
    # Most point cloud models expect a fixed-size input for consistent processing and
    # computational efficiency. The full point cloud (often 10k+ points) can be too large.
    # This step selects the first NUM_POINTS_TEST points after shuffling.
    data = data[0:NUM_POINTS_TEST, :]  # Shape: (NUM_POINTS_TEST, 6)

    # Purpose: To normalize the XYZ coordinates of the point cloud. This involves:
    #   a. Centering: Translating the point cloud so its centroid is at the origin (0,0,0).
    #      This makes the model invariant to the object's absolute position in space.
    #   b. Scaling: Rescaling the point cloud so that all points fit within a unit sphere
    #      (i.e., the furthest point is at a distance of 1 from the origin).
    #      This makes the model invariant to the object's original size or scale.
    # Overall, normalization ensures that the model learns intrinsic shape features rather
    # than being sensitive to arbitrary position and scale, leading to better generalization.
    # It also provides a consistent input range for subsequent voxelization.
    # The user's comment block accurately details further reasons:
    # - Uniform scale for consistent detail capture by the voxel grid.
    # - Translation-invariance.
    # - Stable voxelization by ensuring coordinates are in a predictable range (e.g., [-1,1]).
    data[:, 0:3] = pc_normalize(data[:, 0:3])

    # Purpose: To separate the point cloud data into 3D coordinates (xyz) and
    # surface normal vectors. These are the primary inputs for the model.
    coords_np = data[:, :3]  # xyz positions, Shape: (NUM_POINTS_TEST, 3)
    normals_np = data[:, 3:]  # surface normals, Shape: (NUM_POINTS_TEST, 3)

    # Purpose: To define the features associated with each point. In this case,
    # the features are the concatenation of the 3D coordinates and the 3D normal vectors,
    # resulting in a 6-dimensional feature vector for each point.
    # These features will be aggregated during voxelization.
    feats_np = data  # Shape: (NUM_POINTS_TEST, 6) (contains both coords and normals)

    # --- 2) Prepare inputs for Voxelization ---
    # Purpose: To convert the NumPy arrays into PyTorch tensors and reshape them
    # into the format expected by the voxelization module and PyTorch models.
    #   - `torch.from_numpy()`: Converts NumPy array to PyTorch tensor.
    #   - `.T`: Transposes the arrays from (N, C) to (C, N) because many PyTorch
    #     convolutional/point operations expect the channel dimension (C) before the
    #     number of points/spatial dimension (N).
    #   - `.unsqueeze(0)`: Adds a batch dimension (B=1) at the beginning, making the
    #     shape (B, C, N). PyTorch models typically process data in batches.
    features = torch.from_numpy(feats_np.T).unsqueeze(0)  # Shape: (1, 6, NUM_POINTS_TEST)
    coords = torch.from_numpy(coords_np.T).unsqueeze(0)  # Shape: (1, 3, NUM_POINTS_TEST)

    # --- 3) Voxelize: bin points into a dense R×R×R grid ---
    # Purpose: To initialize the voxelization module.
    #   - `resolution (R)`: Defines the number of bins along each axis of the 3D grid.
    #     The grid will have R*R*R voxels. Higher R means finer granularity.
    #   - `normalize=True`: This flag often indicates that the input `coords` are already
    #     normalized (e.g., to [-1,1] or [0,1]), and the voxelizer will map these
    #     normalized coordinates to voxel indices [0, R-1].
    #   - `eps`: A small epsilon value, typically used for numerical stability in
    #     calculations within the voxelizer, e.g., to prevent division by zero or handle
    #     boundary conditions.
    voxelizer = Voxelization(resolution, normalize=True, eps=1e-6)

    # Purpose: To perform the actual voxelization. This converts the continuous point cloud
    # into a discrete, grid-based representation.
    #   - `features`: The per-point features (coords+normals) to be aggregated.
    #   - `coords`: The per-point 3D coordinates used to determine which voxel each point falls into.
    #   - `averaged_voxel_features`: An (R, R, R) grid where each cell (voxel) contains
    #     the average of the features of all points that fall within that voxel. If a voxel
    #     is empty, its feature vector might be zeros or handled specially. Shape: (B, C_vox, R, R, R)
    #   - `_point2voxel`: (Not used further in this function) A mapping indicating which
    #     voxel each original input point belongs to.
    averaged_voxel_features, _point2voxel = voxelizer(features, coords)

    # Purpose: To calculate and print basic statistics (mean, std, min, max) of the
    # features within the populated voxels. This is a sanity check to understand the
    # distribution of features after voxelization and can help identify issues (e.g.,
    # if all features are zero, or values are unexpectedly large).
    v = averaged_voxel_features
    print("voxel feature stats:", v.mean().item(), v.std().item(), v.min().item(), v.max().item())

    # Purpose: To report the dimensions of the created voxel grid.
    #   - `Bv, C_vox, R, _, _`: Extracts batch size, feature channels per voxel, and resolution.
    #   - `V = R ** 3`: Calculates the total number of voxels in the dense grid.
    # This information confirms the setup and gives an idea of the potential number of
    # "tokens" if a dense transformer were applied to all voxels.
    Bv, C_vox, R_actual, _, _ = averaged_voxel_features.shape  # R_actual should be `resolution`
    V = R_actual ** 3
    print(f"[Test] Voxel grid resolution: {R_actual}³ = {V} tokens (batch size {Bv})")

    # --- 4) Flatten to (B, V, C_vox) ---
    # Purpose: To reshape the 5D voxel feature tensor (B, C_vox, R, R, R) into a 3D
    # tensor (B, V, C_vox). This is a standard format for sequence processing models
    # like Transformers, where:
    #   - `B` is the batch size.
    #   - `V` (total number of voxels, R*R*R) becomes the sequence length (number of tokens).
    #   - `C_vox` is the feature dimension of each voxel token.
    #   - `.view(Bv, C_vox, V)`: Flattens the three spatial dimensions (R,R,R) into one dimension V.
    #   - `.permute(0, 2, 1)`: Swaps the V and C_vox dimensions to get (B, V, C_vox),
    #     making each of the V voxels a token with C_vox features.
    tokens = averaged_voxel_features.view(Bv, C_vox, V).permute(0, 2, 1)  # Shape: (B, V, C_vox)

    # --- 5) Compute continuous voxel-center coords in [-1,1]³ ---
    # Purpose: To calculate the 3D coordinates of the center of each voxel in the
    # normalized space (typically [-1,1] along each axis). These coordinates can be
    # used as positional encodings for the voxel tokens or for calculating distances
    # between voxels (e.g., in k-Nearest Neighbors).
    #   - `torch.arange(R_actual)`: Creates indices [0, 1, ..., R-1].
    #   - `torch.meshgrid()`: Generates a grid of (i,j,k) integer indices for each voxel.
    #   - `.reshape(-1, 3)`: Flattens the grid of indices into a list of (V, 3) coordinates.
    #   - `(grid + 0.5) / R_actual`: Normalizes indices to [0,1) range, pointing to voxel centers.
    #     (e.g., voxel 0 spans [0, 1/R), center is at 0.5/R).
    #   - `* 2.0 - 1.0`: Scales these [0,1) coordinates to the [-1,1) range, matching the
    #     normalized input point cloud space.
    #   - `.unsqueeze(0)`: Adds a batch dimension for consistency.
    idx = torch.arange(R_actual, device=tokens.device)
    grid = torch.stack(torch.meshgrid(idx, idx, idx, indexing='ij'), dim=-1)  # Shape: (R,R,R,3)
    grid = grid.reshape(-1, 3).float()  # Shape: (V,3)
    cont_coords = (grid + 0.5) / R_actual * 2.0 - 1.0  # Normalize to [-1,1]
    cont_coords = cont_coords.unsqueeze(0)  # Shape: (B, V, 3)

    # --- 6) Filter to only non-empty voxels before kNN ---
    # Purpose: To identify and select only those voxels that actually contain data
    # (i.e., had points fall into them during voxelization). Processing empty voxels is
    # computationally wasteful and provides no information. This step makes the
    # subsequent kNN search much more efficient, especially for sparse objects.
    #   - `eps`: A small threshold to consider a voxel "empty" if its feature norm is below it.
    #   - `tokens.norm(dim=-1)`: Calculates the L2 norm of the feature vector for each voxel token.
    #     This gives a measure of the "energy" or "magnitude" of features in each voxel.
    #   - `> eps`: Creates a boolean mask, True for voxels with feature norm greater than eps.
    #   - `.squeeze(0)`: Removes the batch dimension if B=1, resulting in a (V,) mask.
    mask = (tokens.norm(dim=-1) > eps).squeeze(0)  # Shape: (V,) boolean

    # Purpose: To get the actual indices of the non-empty voxels from the boolean mask.
    #   - `.nonzero(as_tuple=False)`: Returns a tensor of indices where the mask is True.
    #   - `.squeeze(1)`: Removes an unnecessary dimension from the output of nonzero.
    non_empty_idxs = mask.nonzero(as_tuple=False).squeeze(1)  # Shape: (V',) where V' is num non-empty

    # Purpose: To count and report how many voxels are non-empty. This indicates the
    # actual number of tokens (V') that will be processed further, highlighting the
    # sparsity of the representation.
    non_empty_count = non_empty_idxs.numel()
    print(f"Non-empty voxels: {non_empty_count} out of {V}")

    # Purpose: A safety check. If no non-empty voxels are found (e.g., due to an
    # empty input point cloud or an issue in prior steps), the function exits to
    # avoid errors in subsequent operations that expect data.
    if non_empty_count == 0:
        print("No data-bearing voxels found. Exiting.")
        return None  # Return None or raise an error

    # Purpose: To create new tensors containing only the features (`tokens_nz`) and
    # continuous center coordinates (`cont_coords_nz`) of the non-empty voxels.
    # This filtering significantly reduces the data size from V total voxels to V'
    # non-empty voxels, making subsequent computations (like kNN) much faster.
    tokens_nz = tokens[:, non_empty_idxs, :]  # Shape: (B, V', C_vox)
    cont_coords_nz = cont_coords[:, non_empty_idxs, :]  # Shape: (B, V', 3)

    # --- 7) Run dynamic-sparse kNN on non-empty voxels ---
    # Purpose: To find the `k` nearest neighbors for each non-empty voxel, using their
    # continuous center coordinates (`cont_coords_nz`) for distance calculation.
    # This establishes a local neighborhood graph among the non-empty voxels.
    # `tokens_nz` might be passed if kNN considers features, but typically it's spatial.
    # The "filter_empty" in a function like `knn_filter_empty` implies it's designed to
    # operate efficiently on this sparse set of non-empty voxel coordinates.
    # The output `neighbors_nz` contains the indices (within the `non_empty_idxs` list)
    # of the k neighbors for each of the V' non-empty voxels.
    neighbors_nz = knn_filter_empty(cont_coords_nz, tokens_nz, k)  # Shape: (B, V', k)

    # Purpose: To construct feature vectors for the "edges" connecting each non-empty
    # voxel to its `k` neighbors. These edge features are crucial for graph neural
    # networks or attention mechanisms as they describe the relationship between
    # connected voxels. Edge features often combine:
    #   - Features of the source voxel (`tokens_nz`).
    #   - Features of the target (neighbor) voxel (obtained by indexing `tokens_nz` with `neighbors_nz`).
    #   - Relative spatial information (e.g., difference in `cont_coords_nz`).
    # Example: `(feat_source, feat_neighbor - feat_source, pos_neighbor - pos_source)`
    edge_feats_nz = build_edge_features(tokens_nz, cont_coords_nz, neighbors_nz)  # Shape: (B, V', k, EdgeFeatDim)

    # Purpose: To pass the constructed edge features through a small neural network
    # (EdgeScorer, likely an MLP). This network learns to assign an "importance score"
    # or "relevance score" to each edge (connection between a voxel and its neighbor).
    # These scores can be used for:
    #   - Weighted aggregation of neighbor information in a GNN.
    #   - Attention weights in a sparse transformer.
    #   - Pruning less important edges from the graph.
    # `in_dim` for EdgeScorer must match the last dimension of `edge_feats_nz`.
    scorer_model = EdgeScorer(in_dim=edge_feats_nz.shape[-1])
    if next(scorer_model.parameters()).is_cuda:  # Check if model is on GPU
        edge_feats_nz = edge_feats_nz.to(next(scorer_model.parameters()).device)
    else:  # If model is on CPU, ensure data is on CPU
        edge_feats_nz = edge_feats_nz.cpu()

    scores_nz = scorer_model(edge_feats_nz)  # Shape: (B, V', k)

    # --- 8) Print shapes for the non-empty subset ---
    # Purpose: For debugging and verification. These print statements display the shapes
    # of the key tensors after filtering for non-empty voxels and performing kNN.
    # This helps confirm that the dimensions are as expected and that `V'` (non-empty
    # voxel count) and `k` (number of neighbors) are correctly reflected.
    print(f"tokens_nz.shape     = {tokens_nz.shape}    # (B={Bv}, V'={tokens_nz.size(1)}, D={C_vox})")
    print(
        f"neighbors_nz.shape  = {neighbors_nz.shape} # (B, V', k={k if non_empty_count > 1 else 0})")  # k can be 0 if V'<=1
    print(f"edge_feats_nz.shape = {edge_feats_nz.shape} # (B, V', k, FeatDim={edge_feats_nz.shape[-1]})")
    print(f"scores_nz.shape     = {scores_nz.shape}\n")  # (B, V', k)

    # --- 9) Sample neighbors for the first non-empty voxel token ---
    # Purpose: A qualitative sanity check. By printing the indices of the first few
    # neighbors (up to 10) of the very first non-empty voxel token, one can get an
    # intuitive idea of whether the kNN search is producing plausible local connections.
    # For example, one might check if these neighbor indices correspond to spatially
    # close voxels if visualized.
    if non_empty_count > 0 and k > 0:  # Ensure there's at least one non-empty voxel and k > 0
        print("Neighbors of non-empty voxel 0:", neighbors_nz[0, 0, :min(10, k)].tolist())

    # Purpose: The function returns the computed edge scores. These scores represent
    # the learned importance or relationship strength for each connection between a
    # non-empty voxel and its k neighbors. They are the primary output of this
    # preprocessing pipeline and would typically be fed into a downstream graph-based
    # learning model (e.g., GNN or sparse Transformer).



if __name__ == "__main__":

    # This makes file paths work for any OS
    txt = os.path.join(DATA_DIR, "airplane", "airplane_0001.txt")
    if not os.path.isfile(txt):
        raise FileNotFoundError(f"Cannot find file: {txt}")
    main(txt)
