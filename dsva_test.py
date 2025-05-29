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
from dsva import knn_filter_empty, build_edge_features, EdgeScorer


def main(txt_path: str, resolution: int = 30, k: int = 100):
    """
    Args:
        txt_path:    Path to a ModelNet40 .txt file with N rows of xyz,nx,ny,nz.
        resolution:  Number of bins per axis for the voxel grid (R).
        k:           Number of nearest neighbors per voxel token.
    """
    # 1) Load the point-cloud + normals: shape = (N, 6)
    # the input files contain more data than is used in practice,
    # and PVT (and most models trained on ModelNet40)
    # typically operate on 1024-point subsamples of each shape.
    data = np.loadtxt(txt_path, delimiter=',', dtype=np.float32)

    # Shuffle the full set of 10k points
    np.random.shuffle(data)

    # Get first 1024 points as in data.py
    data = data[0:NUM_POINTS_TEST, :]

    # Normalize coordinates in point cloud (pc) to unit sphere as in data.py
    """
    Why do we normalize the point cloud to a unit sphere?
    Without this centering-and-scaling step, small objects would be 
    crammed into a few voxels and large ones would overflow the grid, 
    making your voxel transformer’s behavior inconsistent from sample to sample.
    
    Also because: 
    -   Uniform scale – different CAD models can have wildly varying sizes or units. 
        Normalizing to a unit sphere makes sure the same voxel grid resolution 
        captures the same level of detail on every object.
    -   Translation‐invariance – centering at the origin removes any 
        arbitrary offset in the raw coordinates.
    -   Stable voxelization – after scaling into 
        [0,1], you can safely multiply by (R - 1) and round without 
        worrying about negative or out-of-bounds indices.
    """
    data[:, 0:3] = pc_normalize(data[:, 0:3])

    coords_np  = data[:, :3]    # xyz positions
    normals_np = data[:, 3:]    # surface normals

    # Pack all six separate dimensions of data into single 6D vector
    feats_np   = np.concatenate([coords_np, normals_np], axis=1)  # (N,6)

    # 2) Prepare inputs for Voxelization:
    #    features -> (B, C_in, N) = (1, 6, N)
    #    coords   -> (B,   3, N) = (1, 3, N)
    # Set B = 1
    features = torch.from_numpy(feats_np.T).unsqueeze(0)
    coords    = torch.from_numpy(coords_np.T).unsqueeze(0)

    # 3) Voxelize: bin points into a dense R×R×R grid
    voxelizer = Voxelization(resolution, normalize=True, eps=1e-6)
    averaged_voxel_features, _point2voxel = voxelizer(features, coords)

    v = averaged_voxel_features
    print("voxel feature stats:", v.mean().item(), v.std().item(), v.min().item(), v.max().item())
    # Report total voxel tokens
    Bv, C_vox, R, _, _ = averaged_voxel_features.shape
    V = R ** 3
    print(f"[Test] Voxel grid resolution: {resolution}³ = {V} tokens (batch size {Bv})")

    # 4) Flatten to (B, V, C_vox)
    tokens = averaged_voxel_features.view(Bv, C_vox, V).permute(0, 2, 1)

    # 5) Compute continuous voxel-center coords in [-1,1]³
    idx = torch.arange(R)
    grid = torch.stack(torch.meshgrid(idx, idx, idx, indexing='ij'), dim=-1)
    grid = grid.reshape(-1, 3).float()                # (V,3)
    cont_coords = (grid + 0.5) / R * 2.0 - 1.0        # normalize to [-1,1]
    cont_coords = cont_coords.unsqueeze(0)           # (B, V, 3)

    # 6) Filter to only non-empty voxels before kNN
    eps = 1e-6
    mask = (tokens.norm(dim=-1) > eps).squeeze(0)      # (V,) boolean
    non_empty_idxs = mask.nonzero(as_tuple=False).squeeze(1)  # (V',)
    non_empty_count = non_empty_idxs.numel()
    print(f"Non-empty voxels: {non_empty_count} out of {V}")
    if non_empty_count == 0:
        print("No data-bearing voxels found. Exiting.")
        return

    # Subset tokens and coords to V' non-empty voxels
    tokens_nz      = tokens   [:, non_empty_idxs, :]  # (1, V', D)
    cont_coords_nz = cont_coords[:, non_empty_idxs, :]  # (1, V', 3)

    # 7) Run dynamic-sparse kNN on non-empty voxels
    neighbors_nz  = knn_filter_empty(cont_coords_nz, tokens_nz, k)     # (1, V', k)
    edge_feats_nz = build_edge_features(tokens_nz, cont_coords_nz, neighbors_nz)
    scores_nz     = EdgeScorer(in_dim=edge_feats_nz.shape[-1])(edge_feats_nz)

    # 8) Print shapes for the non-empty subset
    print(f"tokens_nz.shape     = {tokens_nz.shape}    # (B={Bv}, V'={tokens_nz.size(1)}, D={C_vox})")
    print(f"neighbors_nz.shape  = {neighbors_nz.shape} # (B, V', k={k})")
    print(f"edge_feats_nz.shape = {edge_feats_nz.shape} # (B, V', k, 2*D+3={edge_feats_nz.shape[-1]})")
    print(f"scores_nz.shape     = {scores_nz.shape}\n")

    # 9) Sample neighbors for the first non-empty voxel token
    print("Neighbors of non-empty voxel 0:", neighbors_nz[0,0,:10].tolist())


if __name__ == "__main__":

    # This makes file paths work for any OS
    txt = os.path.join(DATA_DIR, "airplane", "airplane_0001.txt")
    if not os.path.isfile(txt):
        raise FileNotFoundError(f"Cannot find file: {txt}")
    main(txt)
