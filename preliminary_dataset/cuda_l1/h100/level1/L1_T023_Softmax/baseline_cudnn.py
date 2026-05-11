import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Softmax activation.
    """
    def __init__(self, cudnn_enabled=True, cudnn_benchmark=False, cudnn_deterministic=False):
        super(Model, self).__init__()
        # Set cuDNN backend flags
        torch.backends.cudnn.enabled = cudnn_enabled
        if cudnn_enabled:
            torch.backends.cudnn.benchmark = cudnn_benchmark
            torch.backends.cudnn.deterministic = cudnn_deterministic
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features).

        Returns:
            torch.Tensor: Output tensor with Softmax applied, same shape as input.
        """
        return torch.softmax(x, dim=1)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed