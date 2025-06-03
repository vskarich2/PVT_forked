import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import random


# ─── 1) IMPORT YOUR ACTUAL DATA/MODEL CLASSES HERE ────────────────────────
from data import ScanObjectNNDataset
from model.pvt import pvt

# ─── 2) FUNCTION TO COMPARE TWO CHECKPOINTS ─────────────────────────────
def compare_scanobjectnn_checkpoints(
        ckpt1: str,
        ckpt2: str,
        data_root: str,
        batch_size: int = 32,
        device: str = None
):
    """
    Loads two model checkpoints and returns a list of test‐set indices
    where their predicted labels differ.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate two copies of your real model (here: stub PVTModel)
    model1 = pvt(num_classes=15).to(device)
    model2 = pvt(num_classes=15).to(device)

    # Load their state_dicts (replace these lines with your actual .load_state_dict calls)
    sd1 = torch.load(ckpt1, map_location="cpu")
    sd2 = torch.load(ckpt2, map_location="cpu")
    model1.load_state_dict(sd1)
    model2.load_state_dict(sd2)

    model1.eval()
    model2.eval()

    # Build a DataLoader for ScanObjectNN test split
    test_dataset = ScanObjectNNDataset(root=data_root, split="test", transform=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda")
    )

    disagreements = []
    global_idx = 0

    with torch.no_grad():
        for points, _ in test_loader:
            # points: (B, N_pts, 3)
            points = points.to(device)  # if your model expects (B,3,N_pts), permute here
            logits1 = model1(points)  # (B, num_classes)
            logits2 = model2(points)

            pred1 = logits1.argmax(dim=1).cpu()
            pred2 = logits2.argmax(dim=1).cpu()

            for i in range(pred1.size(0)):
                if int(pred1[i]) != int(pred2[i]):
                    disagreements.append(global_idx + i)

            global_idx += points.size(0)

    return disagreements


# ─── 3) VISUALIZATION & DENSITY METRIC ───────────────────────────────────
def compute_density_metrics(pts: torch.Tensor, k: int = 10):
    """
    Given pts of shape (N_pts, 3), compute:
      • mean distance to each point's k nearest neighbors
      • standard deviation of those k‐neighbor distances
    Returns (mean_of_avg_distances, std_of_avg_distances).
    """
    with torch.no_grad():
        # Compute full N×N distance matrix
        dist_mat = torch.cdist(pts.unsqueeze(0), pts.unsqueeze(0), p=2).squeeze(0)  # shape (N, N)
        # For each point, find its k+1 smallest distances (including the 0 to itself), then skip index 0
        knn_dists, _ = torch.topk(dist_mat, k + 1, largest=False, dim=1)  # (N, k+1)
        knn_dists = knn_dists[:, 1:]  # drop the “self” distance → shape (N, k)
        avg_d = knn_dists.mean(dim=1)  # average distance per point → shape (N,)
        return avg_d.mean().item(), avg_d.std().item()


# ─── 4) RUN COMPARISON AND PLOT ───────────────────────────────────────────
if __name__ == "__main__":
    # 4a) USER CONFIG: replace these with your actual paths
    ckpt_path1 = "/path/to/checkpoint1.pth"
    ckpt_path2 = "/path/to/checkpoint2.pth"
    data_root = "/path/to/ScanObjectNN"
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 4b) Find disagreement indices
    disags = compare_scanobjectnn_checkpoints(
        ckpt1=ckpt_path1,
        ckpt2=ckpt_path2,
        data_root=data_root,
        batch_size=batch_size,
        device=device
    )

    print(f"Found {len(disags)} disagreements; showing up to 3:")

    # 4c) Load full dataset
    full_dataset = ScanObjectNNDataset(root=data_root, split="test", transform=None)

    # 4d) Plot first 3 disagreements (or fewer if there are <3)
    num_to_show = min(3, len(disags))
    fig = plt.figure(figsize=(num_to_show * 4, 4))
    for i in range(num_to_show):
        idx = disags[i]
        pts, label = full_dataset[idx]  # pts: (N_pts, 3)
        ax = fig.add_subplot(1, num_to_show, i + 1, projection="3d")
        ax.scatter(
            pts[:, 0].numpy(),
            pts[:, 1].numpy(),
            pts[:, 2].numpy(),
            s=1,
            c=pts[:, 2].numpy(),
            cmap="viridis"
        )
        ax.set_title(f"Idx {idx}  (Label {label})")
        ax.axis("off")
    plt.show()

    # 4e) Compute density metrics for disagreed examples
    results = []
    for idx in disags[:10]:  # limit to first 10 disagreements
        pts, lbl = full_dataset[idx]
        mean_d, std_d = compute_density_metrics(pts, k=10)
        results.append({
            "index": idx,
            "label": lbl,
            "mean_knn_dist": mean_d,
            "std_knn_dist": std_d,
            "disagreement": True
        })

    # 4f) Also sample a few “agreed” examples and compute the same metric
    all_indices = list(range(len(full_dataset)))
    agreed_indices = [i for i in all_indices if i not in disags]
    sample_agree = random.sample(agreed_indices, min(10, len(agreed_indices)))

    for idx in sample_agree:
        pts, lbl = full_dataset[idx]
        mean_d, std_d = compute_density_metrics(pts, k=10)
        results.append({
            "index": idx,
            "label": lbl,
            "mean_knn_dist": mean_d,
            "std_knn_dist": std_d,
            "disagreement": False
        })

    # 4g) Tabulate results
    df = pd.DataFrame(results)
    print(df)

    # 4h) Optionally, plot a bar‐plot of mean_knn_dist vs. disagreement flag:
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    df_box = df.groupby("disagreement")["mean_knn_dist"].apply(list).to_dict()
    ax2.boxplot([df_box[True], df_box[False]], labels=["Disagreed", "Agreed"])
    ax2.set_ylabel("Avg. distance to 10‐NN")
    ax2.set_title("Point‐Density Metric by Disagreement")
    plt.show()
