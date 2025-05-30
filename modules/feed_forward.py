import torch
import torch.nn as nn

class FeedForward(nn.Module):
    """
    A standard Feed-Forward Network (FFN) block, also known as a Multi-Layer Perceptron (MLP).
    This block is typically used within Transformer architectures after the attention mechanism.
    It applies two linear transformations with a GELU activation and dropout in between.
    """
    def __init__(self, dim, hidden_dim, dropout=0.):
        """
        Initializes the FeedForward module.

        Args:
            dim (int): The input and output feature dimension.
            hidden_dim (int): The dimension of the hidden layer. Typically, hidden_dim > dim
                              to allow the network to learn more complex transformations.
            dropout (float): The dropout rate applied after the GELU activation and
                             after the second linear layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            # First linear transformation: maps input 'dim' to 'hidden_dim'.
            nn.Linear(dim, hidden_dim),
            # Gaussian Error Linear Unit (GELU) activation function.
            # It's a smooth approximation of the ReLU function, often used in Transformers.
            nn.GELU(),
            # Dropout layer: randomly sets a fraction of input units to zero at each update
            # during training time, which helps prevent overfitting.
            nn.Dropout(dropout),
            # Second linear transformation: maps 'hidden_dim' back to 'dim'.
            nn.Linear(hidden_dim, dim),
            # Another dropout layer for regularization.
            nn.Dropout(dropout)
        )

    def forward(self, x):
        """
        Performs the forward pass of the FeedForward block.

        Args:
            x (torch.Tensor): Input tensor, typically of shape (B, N, dim)
                              where N is the sequence length (e.g., number of voxels).

        Returns:
            torch.Tensor: Output tensor of the same shape (B, N, dim).
        """
        return self.net(x)
