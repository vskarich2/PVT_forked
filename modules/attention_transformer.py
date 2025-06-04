import torch
import torch.nn as nn

from modules.dsva.dsva_block import DSVABlock
from modules.sboxblock import Sboxblock
import constants

from modules.dsva.dsva_block import DSVABlockLarge


class Transformer(nn.Module):
    """
    A stack of Sboxblock modules, forming a complete Transformer encoder layer
    for 3D voxel features. It typically alternates between non-shifted and
    shifted window attention blocks.

    AND

    A single DSVA block
    """
    def __init__(
            self,
            out_channels,
            resolution,
            boxsize,
            mlp_dims,
            drop_path1,
            drop_path2,
            args
    ):
        """
        Initializes the Transformer module.

        Args:
            out_channels (int): Feature dimension.
            resolution (int): Voxel grid resolution.
            boxsize (int): Attention window size.
            mlp_dims (int): Hidden dimension for MLPs in Sboxblock.
            drop_path1 (float): DropPath rate for even-indexed Sboxblocks (no shift).
            drop_path2 (float): DropPath rate for odd-indexed Sboxblocks (shifted).
        """
        super().__init__()
        self.args = args
        self.shift = None # Placeholder, actual shift controlled by Sboxblock
        self.depth = 2 # Number of Sboxblock layers in this Transformer module

        # Create a ModuleList of Sboxblock instances.
        # It alternates between non-shifted (shift=None) and shifted (shift=True)
        # attention blocks, and applies different drop_path rates.
        self.blocks = nn.ModuleList([
            Sboxblock(out_channels, resolution, boxsize, mlp_dims,
                      # Shift is None for even blocks (i%2 == 0), True for odd blocks.
                      shift=None if (i % 2 == 0) else True,
                      # DropPath rate alternates between drop_path1 and drop_path2.
                      drop_path=drop_path1 if (i % 2 == 0) else drop_path2)
            for i in range(self.depth)])

        if self.args.large_attn:
            self.dsva_blocks = nn.ModuleList([
                DSVABlockLarge(
                    self.args,
                    out_channels,
                    resolution,
                    # There are two DropPaths; using 1 right now for simplicity .
                    drop_path=drop_path1)
            ])
        else:
            self.dsva_blocks = nn.ModuleList([
                DSVABlock(
                    self.args,
                    out_channels,
                    resolution,
                    mlp_dims,
                    # There are two DropPaths; using 1 right now for simplicity .
                    drop_path=drop_path1)
            ])

    def forward(self, x, non_empty_mask):
        """
        Performs the forward pass through the stack of Sboxblocks
        or the DSVA block.

        Args:
            x (torch.Tensor): Input tensor of shape (B, R^3, C).

        Returns:
            torch.Tensor: Output tensor of the same shape (B, R^3, C).
        """

        if self.args.use_dsva:
            # There is only a single dsva block
            for blk in self.dsva_blocks:
                x = blk(x, non_empty_mask)
            return x
        else:
            # Iterate through each Sboxblock and apply it sequentially.
            for blk in self.blocks:
                x = blk(x)
            return x
