import torch
import torch.nn as nn
import numpy as np

from PVT_forked_repo.PVT_forked.modules.box_attention import rand_bbox
from PVT_forked_repo.PVT_forked.modules.attention_transformer import Transformer


class VoxelEncoder(nn.Module):
    """
    Encodes 3D voxel features. It typically takes a raw voxel grid,
    applies an initial convolution for feature extraction, adds positional
    embeddings, and then processes them through a Transformer block.
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            resolution,
            boxsize,
            mlp_dims,
            drop_path1,
            drop_path2,
            args):
        """
        Initializes the VoxelEncoder.

        Args:
            in_channels (int): Number of input channels to the voxel grid (e.g., from Voxelization).
            out_channels (int): Number of output channels after encoding.
            kernel_size (int): Kernel size for the initial 3D convolution.
            resolution (int): Voxel grid resolution.
            boxsize (int): Attention window size for the Transformer.
            mlp_dims (int): Hidden dimension for MLPs in Transformer blocks.
            drop_path1 (float): DropPath rate for Transformer blocks.
            drop_path2 (float): DropPath rate for Transformer blocks.
        """
        super().__init__()
        self.args = args
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.kernel_size = kernel_size

        # Initial 3D convolutional layer to extract features from the voxel grid.
        # stride=1, padding=kernel_size//2 ensures the output spatial dimensions
        # remain the same as input spatial dimensions (R, R, R).
        self.voxel_emb = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)

        # Layer Normalization applied to the flattened voxel features.
        self.layer_norm = nn.LayerNorm(out_channels)

        # Dropout layer for positional embeddings.
        self.pos_drop = nn.Dropout(p=0.) # Dropout rate is 0.0 by default, effectively disabled.

        # Learnable positional embedding.
        # This tensor adds positional information to each voxel, as Transformers
        # are permutation-invariant without such embeddings.
        # Shape: (1, R^3, C_out) to broadcast across batch dimension.
        self.pos_embedding = nn.Parameter(torch.randn(1, self.resolution ** 3, self.out_channels))

        # Transformer module to process the voxel features.
        # This is where the sbox fixed window attention blocks live AND
        # where DSVA lives.

        self.voxel_Trasformer = Transformer(
            out_channels,
            resolution,
            boxsize,
            mlp_dims,
            drop_path1,
            drop_path2,
            self.args
        )

    def forward(self, inputs, averaged_voxel_tokens):
        """
        Performs the forward pass of the VoxelEncoder.

        Args:
            inputs (torch.Tensor): Input voxel grid of shape (B, C_in, R, R, R).

        Returns:
            torch.Tensor: Encoded voxel features of shape (B, C_out, R, R, R).
        """
        # Apply the initial 3D convolution.
        inputs = self.voxel_emb(inputs) # Shape: (B, C_out, R, R, R)

        # Reshape the 3D voxel features to flattened (B, C_out, R^3) for LayerNorm and Transformer.
        inputs = torch.reshape(inputs, (-1, self.out_channels, self.resolution ** 3))
        # Permute to (B, R^3, C_out) to match the expected input format for LayerNorm and Transformer.
        x = inputs.permute(0, 2, 1) # Shape: (B, R^3, C_out)

        # Apply Layer Normalization.
        x = self.layer_norm(x)

        # Add the learnable positional embedding.
        # This broadcasts the (1, R^3, C_out) pos_embedding to each item in the batch.
        x += self.pos_embedding  # todo: Comment indicating potential future work or reminder.

        # Apply dropout to the features after adding positional embeddings.
        x = self.pos_drop(x)

        # Pass the features through the Transformer where
        # both fixed-window attention and dynamic sparse attention live
        x = self.voxel_Trasformer(x, averaged_voxel_tokens) # Shape: (B, R^3, C_out)

        # Reshape the output from the Transformer back to 3D voxel grid format.
        x = torch.reshape(x, (-1, self.resolution, self.resolution, self.resolution, self.out_channels))
        # Permute dimensions to (B, C_out, R, R, R) to match standard convolutional tensor format.
        x = x.permute(0, 4, 1, 2, 3)
        return x

class SegVoxelEncoder(nn.Module):
    """
    A specialized VoxelEncoder for segmentation tasks, incorporating a CutMix-like
    data augmentation strategy during the forward pass. This helps in regularization
    and improving generalization.
    """
    def __init__(self, in_channels, out_channels, kernel_size, resolution, boxsize, mlp_dims, drop_path1, drop_path2):
        """
        Initializes the SegVoxelEncoder. Parameters are similar to VoxelEncoder.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size for initial convolution.
            resolution (int): Voxel grid resolution.
            boxsize (int): Attention window size.
            mlp_dims (int): Hidden dimension for MLPs.
            drop_path1 (float): DropPath rate.
            drop_path2 (float): DropPath rate.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.resolution = resolution
        self.kernel_size = kernel_size
        # Initial 3D convolutional layer.
        self.voxel_emb = nn.Conv3d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
        # Layer Normalization.
        self.layer_norm = nn.LayerNorm(out_channels)
        # Positional embedding dropout.
        self.pos_drop = nn.Dropout(p=0.)
        # Learnable positional embedding.
        self.pos_embedding = nn.Parameter(torch.randn(1, self.resolution ** 3, self.out_channels))
        # Transformer module.
        self.voxel_Trasformer = Transformer(out_channels, resolution, boxsize, mlp_dims, drop_path1, drop_path2)
        # Beta parameter for the Beta distribution used in rand_bbox for CutMix.
        self.beta = 1.

    def forward(self, inputs):
        """
        Performs the forward pass of the SegVoxelEncoder, including CutMix augmentation.

        Args:
            inputs (torch.Tensor): Input voxel grid of shape (B, C_in, R, R, R).

        Returns:
            torch.Tensor: Encoded voxel features of shape (B, C_out, R, R, R),
                          potentially with CutMix applied.
        """
        # Apply initial 3D convolution.
        inputs = self.voxel_emb(inputs) # Shape: (B, C_out, R, R, R)

        # --- CutMix Data Augmentation ---
        # Generate a lambda value from a Beta distribution.
        lam = np.random.beta(self.beta, self.beta)
        # Generate random bounding box coordinates for the cutout region.
        bbx1, bby1, bbx2, bby2, bbz1, bbz2 = rand_bbox(inputs.size(), lam)

        # Create a clone of the input tensor to perform CutMix.
        temp_x = inputs.clone()
        # Apply CutMix: The region defined by the bounding box in the current batch
        # is replaced with the corresponding region from a flipped version of the batch.
        # inputs.flip(0) reverses the batch order.
        temp_x[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2] = inputs.flip(0)[:, :, bbx1:bbx2, bby1:bby2, bbz1:bbz2]
        # Update inputs with the CutMixed tensor.
        inputs = temp_x

        # Reshape to flattened (B, C_out, R^3) for LayerNorm and Transformer.
        inputs = torch.reshape(inputs, (-1, self.out_channels, self.resolution ** 3))
        # Permute to (B, R^3, C_out) for LayerNorm and Transformer.
        x = inputs.permute(0, 2, 1)

        # Apply Layer Normalization.
        x = self.layer_norm(x)
        # Add positional embedding.
        x += self.pos_embedding  # todo
        # Apply positional embedding dropout.
        x = self.pos_drop(x)
        # Pass through the Transformer.
        x = self.voxel_Trasformer(x) # Shape: (B, R^3, C_out)

        # Reshape back to 3D voxel grid format.
        x = torch.reshape(x, (-1, self.resolution, self.resolution, self.resolution, self.out_channels))

        # --- Apply CutMix to the output as well (or undo the input CutMix) ---
        # This step seems to apply the same CutMix operation to the output of the encoder.
        # This is a bit unusual for standard CutMix, which typically modifies inputs and labels.
        # It might be intended to ensure consistency if the model is expected to handle
        # CutMixed inputs throughout its layers, or to apply a similar regularization
        # at the output level.
        temp_x = x.clone()
        temp_x[:, bbx1:bbx2, bby1:bby2, bbz1:bbz2, :] = x.flip(0)[:, bbx1:bbx2, bby1:bby2, bbz1:bbz2, :]
        x = temp_x

        # Permute dimensions to (B, C_out, R, R, R) for consistency.
        x = x.permute(0, 4, 1, 2, 3)
        return x

