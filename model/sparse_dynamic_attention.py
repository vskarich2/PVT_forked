import torch


def get_nn(coords, k):
    # coords: (B, N, 3)
    # k: number of neighbors

    dists = torch.cdist(coords, coords)  # (B, N, N)
    _, nn_idx = dists.topk(k + 1, dim=-1, largest=False)

    # first neighbor is itself (distance=0), so slice it off if you want k true neighbors
    nn_idx = nn_idx[:, :, 1:]  # (B, N, k)

