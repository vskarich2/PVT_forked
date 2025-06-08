import torch
import torch.nn as nn
from torch.nn import functional as F

from model.utils import create_pointnet_components

__all__ = ['pvt']

class pvt(nn.Module):
    # Define four blocks: (output channels, number of repeats, resolution divisor)
    blocks = (
        (64, 1, 30),  # one PVTConv block → 64-dim features on a 30³ grid
        (128, 2, 15),  # two PVTConv blocks → 128-dim features on a 15³ grid
        (512, 1, None),
        (1024, 1, None),
    )

    def __init__(
        self,
        args,
        num_classes: int = 40,
        width_multiplier: float = 1,
        voxel_resolution_multiplier: float = 1,
    ):
        super().__init__()
        self.args = args
        # Input channels: x,y,z + normal_x, normal_y, normal_z
        self.in_channels = 6  # features shape: (B, 6, N)

        # Save gradients and activations for saliency maps
        self._attn_acts = []
        self._attn_grads = []

        # Create the sequence of PVTConv blocks and compute output channel counts
        layers, channels_point, concat_channels_point = create_pointnet_components(
            args=self.args,
            blocks=self.blocks,
            in_channels=self.in_channels,
            normalize=False,
            width_multiplier=width_multiplier,
            voxel_resolution_multiplier=voxel_resolution_multiplier,
            model='PVTConv'
        )
        # layers: list of PVTConv modules
        # channels_point: sum of output channels of each block
        # concat_channels_point: channels from concatenation of intermediate features

        # Wrap PVTConv modules in a ModuleList for iteration
        self.point_features = nn.ModuleList(layers)  # length = len(self.blocks)

        # After concatenating all features, fuse down to 1024 channels via 1D conv
        # Input channels = channels_point + concat_channels_point + channels_point
        self.conv_fuse = nn.Sequential(
            nn.Conv1d(
                channels_point + concat_channels_point + channels_point,
                1024,
                kernel_size=1,
                bias=False
            ),
            nn.BatchNorm1d(1024)
        )

        # Classification head: two-layer MLP with dropout
        self.linear1 = nn.Linear(1024, 512)
        self.dp1     = nn.Dropout(0.5)
        self.linear2 = nn.Linear(512, 256)
        self.dp2     = nn.Dropout(0.5)
        self.linear3 = nn.Linear(256, num_classes)  # output logits shape: (B, num_classes)
        self.bn1     = nn.BatchNorm1d(512)
        self.bn2     = nn.BatchNorm1d(256)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features: (B, 6, N) where N = number of points
        # This is the point cloud grid
        B, C_in, N = features.shape

        # Extract coordinates (first 3 channels) for point-voxel operations
        coords = features[:, :3, :]  # shape: (B, 3, N)

        # Will collect outputs from each PVTConv block
        out_features_list = []  # list of tensors of shape (B, C_i, N)

        # 1) Apply each PVTConv block sequentially
        #    Each block returns updated features and unchanged coords
        for block in self.point_features:
            features, _ = block((features, coords))  # features: (B, C_i, N)
            print("PVTConv 9.2")
            out_features_list.append(features)
            print("PVTConv 9.5")

        if self.args.scanobject_compare:
            print("PVTConv 10")

        # 2) Append per-point max and mean statistics
        #    a) Max over channel dim -> shape (B, C_last, 1) -> repeat to (B, C_last, N)
        max_feat = features.max(dim=-1, keepdim=True).values  # (B, C_last, 1)
        max_feat = max_feat.repeat(1, 1, N)                  # (B, C_last, N)
        out_features_list.append(max_feat)
        if self.args.scanobject_compare:
            print("PVTConv 11")

        #    b) Mean over channel dim -> (B, 1, N) then expand channels
        mean_feat = features.mean(dim=-1, keepdim=True)      # (B, C_last, 1)
        mean_feat = mean_feat.repeat(1, 1, N)                # (B, C_last, N)
        out_features_list.append(mean_feat)

        if self.args.scanobject_compare:
            print("PVTConv 12")

        # 3) Concatenate all collected features along channel axis
        #    result shape: (B, total_channels, N)
        features = torch.cat(out_features_list, dim=1)
        if self.args.scanobject_compare:
            print("PVTConv 13")
            print(features.shape)

        # 4) Fuse high-dimensional features down to 1024 channels
        features = self.conv_fuse(features)
        if self.args.scanobject_compare:
            print("PVTConv 14")
        features = F.leaky_relu(self.conv_fuse(features))    # (B, 1024, N)

        if self.args.scanobject_compare:
            print("PVTConv 15")
        # 5) Global pooling to get a per-cloud descriptor
        features = F.adaptive_max_pool1d(features, 1)  # (B, 1024, 1)
        features = features.view(B, -1)                # (B, 1024)

        # 6) Classification MLP with BatchNorm, activation, dropout
        features = F.leaky_relu(self.bn1(self.linear1(features)))  # (B, 512)
        features = self.dp1(features)
        features = F.leaky_relu(self.bn2(self.linear2(features)))  # (B, 256)
        features = self.dp2(features)

        # 7) Final linear layer -> logits (B, num_classes)
        logits = self.linear3(features)
        return logits
