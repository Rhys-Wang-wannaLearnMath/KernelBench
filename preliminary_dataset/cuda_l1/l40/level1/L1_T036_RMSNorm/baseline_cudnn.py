import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs RMS Normalization.
    """
    def __init__(self, num_features: int, eps: float = 1e-5):
        """
        Initializes the RMSNorm layer.

        Args:
            num_features (int): Number of features in the input tensor.
            eps (float, optional): A small value added to the denominator to avoid division by zero. Defaults to 1e-5.
        """
        super(Model, self).__init__()
        self.num_features = num_features
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies RMS Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, *).

        Returns:
            torch.Tensor: Output tensor with RMS Normalization applied, same shape as input.
        """
        # Use the cudnn.flags context manager to enable backend optimizations for this block.
        # benchmark=True enables the auto-tuner to find the best algorithm for the hardware.
        # Although RMSNorm is not a direct cuDNN-accelerated op like convolution,
        # this is the standard pattern for enabling cuDNN flags for a model's forward pass.
        with torch.backends.cudnn.flags(enabled=True, benchmark=True, deterministic=False):
            # Calculate the RMS along the feature dimension
            rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)

            # Normalize the input by dividing by the RMS
            return x / rms

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [features]