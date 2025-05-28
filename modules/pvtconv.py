import torch
import torch.nn as nn
import modules.functional as F
from modules.voxelization import Voxelization
from modules.shared_transformer import SharedTransformer
from modules.se import SE3d
from timm.models.layers import DropPath
import numpy as np

__all__ = ['PVTConv','PartPVTConv','SemPVTConv']

# Utility to generate a random 3D bounding box for data augmentation
def rand_bbox(size, lam):
    # size: tensor shape (B, C, D, H, W)
    x = size[2]  # depth dimension
    y = size[3]  # height dimension
    z = size[4]  # width dimension
    # determine box size via lambda
    cut_rat = np.sqrt(1. - lam)
    cut_x = int(x * cut_rat)
    cut_y = int(y * cut_rat)
    cut_z = int(z * cut_rat)
    # random center point
    cx = np.random.randint(x)
    cy = np.random.randint(y)
    cz = np.random.randint(z)
    # compute box edges, clipped to volume bounds
    bbx1 = np.clip(cx - cut_x // 2, 0, x)
    bbx2 = np.clip(cx + cut_x // 2, 0, x)
    bby1 = np.clip(cy - cut_y // 2, 0, y)
    bby2 = np.clip(cy + cut_y // 2, 0, y)
    bbz1 = np.clip(cz - cut_z // 2, 0, z)
    bbz2 = np.clip(cz + cut_z // 2, 0, z)
    return bbx1, bby1, bbx2, bby2, bbz1, bbz2


# Simple two-layer feed-forward module with GELU activation
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),  # expand
            nn.GELU(),                   # non-linear
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),  # project back
            nn.Dropout(dropout)
        )
    def forward(self, x):
        # x: (B, N, dim) → output same shape
        return self.net(x)


# Partition a 5D tensor of shape (B, R, R, R, C) into smaller cubes of size box_size
def box_partition(x, box_size):
    b, R, _, _, C = x.shape
    # reshape to group cubes: (B, R/bs, bs, R/bs, bs, R/bs, bs, C)
    x = x.view(
        b,
        R//box_size, box_size,
        R//box_size, box_size,
        R//box_size, box_size,
        C
    )
    # permute to gather each cube into batch dimension
    boxes = x.permute(0,1,3,5,2,4,6,7)
    # flatten batch and cube dims → (B * num_boxes, bs, bs, bs, C)
    boxes = boxes.contiguous().view(-1, box_size, box_size, box_size, C)
    return boxes

# Reverse the box partition to reconstruct the original tensor
def box_reverse(boxes, box_size, resolution):
    # compute batch by dividing total boxes by cubes per volume
    b = int(boxes.shape[0] / ((resolution//box_size)**3))
    # reshape back to (B, R/bs, R/bs, R/bs, bs, bs, bs, C)
    x = boxes.view(
        b,
        resolution//box_size, resolution//box_size, resolution//box_size,
        box_size, box_size, box_size,
        -1
    )
    # permute to (B, R/bs, bs, R/bs, bs, R/bs, bs, C)
    x = x.permute(0,1,4,2,5,3,6,7)
    # collapse to (B, R, R, R, C)
    x = x.contiguous().view(b, resolution, resolution, resolution, -1)
    return x

# Window-based multi-head self-attention inside each local cube
class BoxAttention(nn.Module):
    def __init__(self, dim, box_size, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.box_size = box_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        # relative position bias table: (num_rel_positions, heads)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((3*box_size-1)**3, num_heads)
        )
        # precompute index map from pairwise offsets to bias table indices
        coords = torch.stack(torch.meshgrid(
            [torch.arange(box_size) for _ in range(3)], indexing='ij'
        ))  # (3, bs, bs, bs)
        coords_flat = coords.flatten(1)       # (3, bs^3)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (3, bs^3, bs^3)
        rel = rel.permute(1,2,0).contiguous()  # (bs^3, bs^3, 3)
        # shift to positive and compute a single index
        rel += box_size - 1
        rel[...,0] *= (2*box_size-1)*(2*box_size-1)
        rel[...,1] *= (2*box_size-1)
        idx = rel.sum(-1)  # (bs^3, bs^3)
        self.register_buffer('relative_position_index', idx)
        # qkv projection
        self.qkv = nn.Linear(dim, dim*3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x, mask=None):
        B_, N, C = x.shape  # N=box_size^3
        # project and reshape into heads
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads)
        q,k,v = qkv.permute(2,0,3,1,4)
        q = q * self.scale
        # compute attention logits
        attn = q @ k.transpose(-2,-1)  # (B_, heads, N, N)
        # add relative bias
        bias = self.relative_position_bias_table[self.relative_position_index.flatten()]
        bias = bias.view(N, N, -1).permute(2,0,1)
        attn = attn + bias.unsqueeze(0)
        # optional masking for shifted windows
        if mask is not None:
            Bsplit = B_ // mask.shape[0]
            attn = attn.view(Bsplit, mask.shape[0], self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        # attend to v
        x = (attn @ v).transpose(1,2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    def extra_repr(self):
        return f'dim={self.dim}, box_size={self.box_size}, num_heads={self.num_heads}'

# Shifted window block with residual, LayerNorm, and MLP
class Sboxblock(nn.Module):
    """
    By rolling the feature map before you slice it into cubic windows,
    you ensure that the window boundaries “move” by half a window each
    alternating block. Then, after you do attention on those windows,
    you roll it back so that everything realigns.

    This alternating “no shift → shift → no shift → shift…” pattern is
    exactly how Swin/shifted‐window Transformers avoid blind spots at
    window borders, and your Sboxblock implements it on your 3D voxel grid.
    """

    def __init__(
        self, out_channels, resolution, boxsize,
        mlp_dims, shift=True, drop_path=0.
    ):
        super().__init__()
        self.resolution = resolution
        self.box_size = boxsize
        self.shift_size = boxsize//2 if shift else 0
        self.attn = BoxAttention(out_channels, boxsize, num_heads=4)
        # build mask for shifted windows
        if self.shift_size > 0:
            img_mask = torch.zeros((1, resolution, resolution, resolution, 1))
            slices = (slice(0, -boxsize), slice(-boxsize, -self.shift_size), slice(-self.shift_size, None))
            cnt=0
            for x in slices:
                for y in slices:
                    for z in slices:
                        img_mask[:, x, y, z, :] = cnt
                        cnt += 1
            mask_boxes = box_partition(img_mask, boxsize).view(-1, boxsize**3)
            attn_mask = mask_boxes.unsqueeze(1) - mask_boxes.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask!=0, float(-100.0)).masked_fill(attn_mask==0, 0.0)
        else:
            attn_mask = None
        self.register_buffer('attn_mask', attn_mask)
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.mlp = FeedForward(out_channels, mlp_dims)
        self.drop_path = DropPath(drop_path)
    def forward(self, x):
        # x: (B, N, C)
        shortcut = x
        B_ = x.shape[0]
        x = self.norm1(x)
        # reshape to 3D volume to apply shifting & window partition
        x_vol = x.view(B_, self.resolution, self.resolution, self.resolution, -1)
        # cyclic shift
        if self.shift_size>0:
            x_shift = torch.roll(x_vol, shifts=(-self.shift_size,)*3, dims=(1,2,3))
        else:
            x_shift = x_vol
        # partition into windows
        windows = box_partition(x_shift, self.box_size).view(-1, self.box_size**3, self.out_channels)
        # window attention with mask
        attn_windows = self.attn(windows, mask=self.attn_mask)
        # merge windows back into volume
        x_merge = box_reverse(attn_windows, self.box_size, self.resolution)
        # reverse cyclic shift
        if self.shift_size>0:
            x_merge = torch.roll(x_merge, shifts=(self.shift_size,)*3, dims=(1,2,3))
        # flatten back to tokens
        x = x_merge.view(B_, -1, self.out_channels)
        # residual + drop_path
        x = shortcut + 0.5 * self.drop_path(x)
        # MLP block
        x = x + 0.5 * self.drop_path(self.mlp(self.norm2(x)))
        return x

# Transformer wrapper with two shifted-window blocks
class Transformer(nn.Module):
    def __init__(self, out_channels, resolution, boxsize, mlp_dims, drop1, drop2):
        super().__init__()
        self.blocks = nn.ModuleList([
            Sboxblock(
                out_channels, resolution, boxsize,
                mlp_dims, shift=(i%2==1), drop_path=(drop1 if i%2==0 else drop2)
            ) for i in range(2)
        ])
    def forward(self, x):
        # x: (B, R^3, C)
        for blk in self.blocks:
            x = blk(x)
        return x


class VoxelEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,      # C_pt: number of input point-feature channels (e.g., 6)
        out_channels: int,     # d_model: desired embedding dimension per voxel
        kernel_size: int,      # size of the 3D conv kernel (e.g., 3)
        resolution: int,       # R: grid resolution per axis
        boxsize,               # unused here, passed to Transformer
        mlp_dims,              # dims for the internal MLP in Transformer
        drop_path1, drop_path2 # dropout rates for Transformer
    ):
        super().__init__()
        self.in_channels  = in_channels     # store for reference
        self.out_channels = out_channels    # store for reference
        self.resolution   = resolution      # store grid size R
        self.kernel_size  = kernel_size

        # 1) 3D convolution to embed raw voxel features into d_model channels
        #    Input:  (B, C_pt, R, R, R)
        #    Output: (B, d_model, R, R, R)
        self.voxel_emb = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=kernel_size // 2
        )

        # 2) LayerNorm over the feature dimension of each token sequence
        #    Will be applied to tensors of shape (B, R^3, d_model)
        self.layer_norm = nn.LayerNorm(out_channels)

        # 3) Dropout for the positional embeddings
        self.pos_drop = nn.Dropout(p=0.0)

        # 4) Learnable positional embeddings for the R^3 tokens
        #    Shape: (1, R^3, d_model), broadcast across batch
        self.pos_embedding = nn.Parameter(
            torch.randn(1, resolution ** 3, out_channels)
        )

        # 5) The actual Transformer block over the voxel tokens
        #    Expects input shape (B, R^3, d_model) → output same shape
        self.voxel_Transformer = Transformer(
            out_channels, resolution, boxsize,
            mlp_dims, drop_path1, drop_path2
        )

    def forward(self, inputs):
        # inputs:  (B, C_pt, R, R, R) from voxelization
        # 1) Project raw voxel features into embedding space
        x = self.voxel_emb(inputs)
        #    x shape: (B, d_model, R, R, R)

        # 2) Flatten the 3D grid into a sequence of R^3 tokens
        #    First reshape to (B, d_model, R^3)
        x = x.view(-1, self.out_channels, self.resolution ** 3)
        #    Then permute to (B, R^3, d_model)
        x = x.permute(0, 2, 1)

        # 3) Normalize each token's features
        x = self.layer_norm(x)
        #    x shape remains (B, R^3, d_model)

        # 4) Add learned positional embeddings
        x = x + self.pos_embedding
        #    broadcasting over batch dimension

        # 5) Optional dropout on positions
        x = self.pos_drop(x)

        # 6) Run through the Transformer block
        #    input & output both (B, R^3, d_model)
        x = self.voxel_Transformer(x)

        # 7) Reshape sequence back into 3D grid
        #    First to (B, R^3, d_model) → (B, R, R, R, d_model)
        x = x.view(-1, self.resolution, self.resolution, self.resolution, self.out_channels)

        # 8) Permute to standard Conv3D layout: (B, d_model, R, R, R)
        x = x.permute(0, 4, 1, 2, 3)

        return x  # final voxel-encoded features for the next stage

class SegVoxelEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution,boxsize,mlp_dims,drop_path1,drop_path2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.kernel_size = kernel_size
        self.voxel_emb = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.pos_drop = nn.Dropout(p=0.)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.resolution ** 3, self.out_channels))
        self.voxel_Transformer = Transformer(out_channels, resolution, boxsize,mlp_dims,drop_path1,drop_path2)
        self.beta = 1.

    def forward(self, inputs):
        inputs = self.voxel_emb(inputs)
        lam = np.random.beta(self.beta, self.beta)
        bbx1, bby1, bbx2, bby2, bbz1, bbz2 = rand_bbox(inputs.size(), lam)
        temp_x = inputs.clone()
        temp_x[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = inputs.flip(0)[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]
        inputs = temp_x
        inputs = torch.reshape(inputs, (-1, self.out_channels, self.resolution ** 3))
        x = inputs.permute(0, 2, 1)
        x = self.layer_norm(x)
        x += self.pos_embedding  # todo
        x = self.pos_drop(x)
        x = self.voxel_Transformer(x)
        x = torch.reshape(x, (-1, self.resolution, self.resolution, self.resolution, self.out_channels))
        temp_x = x.clone()
        temp_x[:, bbx1:bbx2, bby1:bby2, bbz1:bbz2, :] = x.flip(0)[:, bbx1:bbx2, bby1:bby2, bbz1:bbz2, :]
        x = temp_x
        x = x.permute(0, 4, 1, 2, 3)
        return x

class PVTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.boxsize = 3
        self.mlp_dims = out_channels
        self.drop_path1 = 0.1
        self.drop_path2 = 0.2

        # Converts point-clouds to voxels
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)

        # Encodes voxels into a feature vector
        # by applying 3D convolutions + local-window self-attention.
        self.voxel_encoder = VoxelEncoder(in_channels, out_channels, kernel_size, resolution,self.boxsize,
                                          self.mlp_dims, self.drop_path1, self.drop_path2)

        # Squeeze‐and‐Excitation: channel-wise gating that re-weights.
        # each channel of the voxel features adaptively.
        self.SE = SE3d(out_channels)

        # What does this do?
        self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        # Unpack inputs:
        # features: (B, C_in, N) point-wise features (e.g., xyz normals embedded)
        # coords:   (B, 3, N) original point coordinates
        features, coords = inputs

        # -------------------------------
        # 1) Voxel path: convert points → voxels → back to points
        # -------------------------------
        """
        
        1. `coords` (raw points):
        - Shape: `(B, 3, N)` where B is batch size and N is number of points
        - Contains the raw 3D coordinates (x,y,z) of each point in the point cloud
        - These are the actual spatial positions of points in 3D space
        
        2. `features` (point features):
        - Shape: `(B, C_in, N)` where C_in is the number of input channels
        - Contains per-point feature vectors that can include:
          - Geometric features (like surface normals)
          - Learned embeddings from previous layers
          - Any other point-wise attributes (like color, intensity, etc.)
        - These features represent richer information about each point beyond just its position
        - Specifically for our model, C_in includes both xyz (3) and surface normals (3)
        
        The key difference is that:
        - `coords` tells you WHERE each point is in 3D space
        - `features` tells you WHAT each point represents or what properties it has
        
        When `self.voxelization(features, coords)` is called, it:
        1. Uses `coords` to determine which voxel grid cell each point belongs to
        2. For points that fall into the same voxel, it averages their `features` to create a feature vector for that voxel
        3. Returns:
           - `voxel_features`: `(B, C_in, R, R, R)` - a dense 3D grid where each voxel contains averaged features
           Note that it is perhaps more intuitive to think about the voxel grid
           as having shape (B, R, R, R, C_in). 
           - `voxel_coords`: `(B, N, 3)` - mapping between original points and their corresponding voxel indices
        
        This is a key step in the PVT (Point-Voxel Transformer) architecture as it bridges between the continuous point cloud representation and a discrete voxel grid representation, while preserving the rich feature information of each point.

        
        """
        # The R^3 voxel grid is the total number of voxels we have at our disposal.
        # This is the number of bins into which we sort the raw points.
        voxel_features, voxel_coords = self.voxelization(features, coords)

        # b) Voxel encoder: apply 3D convolutions + local-window self-attention
        #    treats the R³ grid as tokens, embeds them, and aggregates local context
        #    output stays (B, C_voxel, R, R, R)
        voxel_features = self.voxel_encoder(voxel_features)

        # c) Squeeze‐and‐Excitation: channel-wise gating
        #    re-weights each channel of the voxel features adaptively
        voxel_features = self.SE(voxel_features)

        # d) Trilinear devoxelization: interpolate voxel grid back to N points
        #    uses voxel_coords to map each original point to its 8 neighboring voxels
        #    and returns (B, C_voxel, N) point features via smooth interpolation
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)

        # -------------------------------
        # 2) Point path: pure point‐cloud self‐attention
        # -------------------------------

        # a) Rearrange coords to (B, N, 3) for pairwise diff
        pos = coords.permute(0, 2, 1)
        # b) Compute pairwise relative positions:
        #    rel_pos[i,j] = sum over xyz of (pos[i] - pos[j])
        #    This yields a (B, N, N) matrix of scalar biases
        rel_pos = pos[:, :, None, :] - pos[:, None, :, :]
        rel_pos = rel_pos.sum(dim=-1)

        # -------------------------------
        # 3) Fuse voxel + point outputs
        # -------------------------------

        # Element-wise sum of the two parallel paths:
        # voxel_features: (B, C_voxel, N)
        # point_out:      (B, C_voxel, N)
        point_out = self.point_features(features, rel_pos)

        # Return the fused per-point features and unchanged coords
        fused_features = voxel_features + point_out
        return fused_features, coords

class PartPVTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.boxsize = 3
        self.mlp_dims = out_channels
        self.drop_path1 = 0.1
        self.drop_path2 = 0.1
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        self.voxel_encoder = VoxelEncoder(in_channels, out_channels, kernel_size, resolution,self.boxsize,
                                             self.mlp_dims,self.drop_path1,self.drop_path2)
        self.SE = SE3d(out_channels)
        self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_encoder(voxel_features)
        voxel_features = self.SE(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords

class SemPVTConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, resolution, normalize=True, eps=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.resolution = resolution
        self.boxsize = 4
        self.mlp_dims = out_channels*4
        self.drop_path1 = 0.
        self.drop_path2 = 0.1
        self.voxelization = Voxelization(resolution, normalize=normalize, eps=eps)
        self.voxel_encoder = SegVoxelEncoder(in_channels, out_channels, kernel_size, resolution,self.boxsize,
                                             self.mlp_dims,self.drop_path1,self.drop_path2)
        self.SE = SE3d(out_channels)
        self.point_features = SharedTransformer(in_channels, out_channels)

    def forward(self, inputs):
        features, coords = inputs
        voxel_features, voxel_coords = self.voxelization(features, coords)
        voxel_features = self.voxel_encoder(voxel_features)
        voxel_features = self.SE(voxel_features)
        voxel_features = F.trilinear_devoxelize(voxel_features, voxel_coords, self.resolution, self.training)
        fused_features = voxel_features + self.point_features(features)
        return fused_features, coords
