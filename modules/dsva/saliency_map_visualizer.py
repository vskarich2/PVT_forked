import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plotting

from PVT_forked_repo.PVT_forked.modules.dsva.dsva_saliency import DynamicSparseSaliency


def visualize_voxel_saliency(
    voxel_coords: np.ndarray,
    saliency_vals: np.ndarray,
    points: np.ndarray = None,
    cmap: str = 'hot',
    point_size: float = 1.0,
    voxel_size: float = 50.0,
    elev: float = 30.0,
    azim: float = 45.0,
    figsize: tuple = (8, 6),
    title: str = "Voxel Saliency"
):
    """
    Visualize per-voxel saliency in 3D.

    Args:
        voxel_coords (np.ndarray): shape (M, 3), the 3D coordinates of each voxel center.
        saliency_vals (np.ndarray): shape (M,), saliency scores normalized to [0,1].
        points (np.ndarray, optional): shape (N, 3), raw point cloud for context (plotted in gray).
                                        If None, only voxel centers are shown.
        cmap (str): Matplotlib colormap name for mapping saliency → color (default 'hot').
        point_size (float): size of raw points in the background.
        voxel_size (float): size of the voxel‐center scatter markers.
        elev (float): elevation angle (in degrees) for 3D view.
        azim (float): azimuth angle (in degrees) for 3D view.
        figsize (tuple): figure size for matplotlib.
        title (str): title of the plot.

    Returns:
        fig, ax: Matplotlib figure and 3D axis objects, so you can adjust or save externally.
    """
    assert voxel_coords.ndim == 2 and voxel_coords.shape[1] == 3, "voxel_coords must be (M,3)"
    assert saliency_vals.ndim == 1 and saliency_vals.shape[0] == voxel_coords.shape[0], \
        "saliency_vals must be shape (M,)"
    if points is not None:
        assert points.ndim == 2 and points.shape[1] == 3, "points must be (N,3) or None"

    # Convert to numpy if they are tensors
    if hasattr(voxel_coords, 'detach'):
        voxel_coords = voxel_coords.detach().cpu().numpy()
    if hasattr(saliency_vals, 'detach'):
        saliency_vals = saliency_vals.detach().cpu().numpy()
    if points is not None and hasattr(points, 'detach'):
        points = points.detach().cpu().numpy()

    # Create the figure and 3D axes
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title, fontsize=12)

    # Optionally plot raw points in light gray, small size
    if points is not None:
        ax.scatter(
            points[:, 0], points[:, 1], points[:, 2],
            c='lightgray', s=point_size, alpha=0.2, linewidth=0
        )

    # Map saliency to colors
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(saliency_vals)  # RGBA for each voxel

    # Plot voxel centers, colored by saliency
    sc = ax.scatter(
        voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2],
        c=colors, s=voxel_size, depthshade=True, linewidth=0
    )

    # Optionally add a colorbar
    mappable = plt.cm.ScalarMappable(cmap=cmap_obj)
    mappable.set_array(saliency_vals)
    cbar = fig.colorbar(mappable, shrink=0.5, aspect=10, pad=0.1)
    cbar.set_label('Saliency', rotation=270, labelpad=12)

    # Adjust view angle
    ax.view_init(elev=elev, azim=azim)

    # Label axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Tight layout
    plt.tight_layout()
    return fig, ax


import torch

# Suppose `voxel_coords` is a tensor of shape [M, 3] (voxel centers)
# and `voxel_feats` is [M, D] (features),
# `raw_points` is [N, 3], and your model is `dynamic_sparse_model`.

# 1. Compute all three saliency maps (attention, gradient, occlusion)
saliency_helper = DynamicSparseSaliency(dynamic_sparse_model, device=torch.device('cuda'))
voxel_feats = voxel_feats.to(torch.device('cuda'))
logits = dynamic_sparse_model(voxel_feats.unsqueeze(0))
pred_class = logits.argmax(dim=1).item()
saliency_maps = saliency_helper.compute_all_saliency(voxel_feats, pred_class)

# 2. Convert voxel_coords (tensor) and points to numpy
voxel_coords_np = voxel_coords.detach().cpu().numpy()  # shape (M, 3)
points_np = raw_points.detach().cpu().numpy()  # shape (N, 3)

# 3. Visualize each saliency map
attn_saliency = saliency_maps['attention_flow']  # Tensor [M]
grad_saliency = saliency_maps['gradient_sensitivity']  # Tensor [M]
occ_saliency  = saliency_maps['occlusion']  # Tensor [M]

# Plot Attention Flow
fig1, ax1 = visualize_voxel_saliency(
    voxel_coords_np, attn_saliency,
    points=points_np, cmap='hot', voxel_size=50.0,
    title="Attention Flow Saliency"
)
plt.show()

# Plot Gradient Sensitivity
fig2, ax2 = visualize_voxel_saliency(
    voxel_coords_np, grad_saliency,
    points=points_np, cmap='viridis', voxel_size=50.0,
    title="Gradient‐Sensitivity Saliency"
)
plt.show()

# Plot Occlusion‐Based Saliency
fig3, ax3 = visualize_voxel_saliency(
    voxel_coords_np, occ_saliency,
    points=points_np, cmap='magma', voxel_size=50.0,
    title="Occlusion‐Based Saliency"
)
plt.show()
