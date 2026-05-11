import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a LogSoftmax activation.
    """
    def __init__(self, dim: int = 1):
        super(Model, self).__init__()
        self.dim = dim
        self.cudnn_enable = True
        self.cudnn_benchmark = False
        self.cudnn_deterministic = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies LogSoftmax activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, dim).

        Returns:
            torch.Tensor: Output tensor with LogSoftmax applied, same shape as input.
        """
        with torch.backends.cudnn.flags(
            enabled=self.cudnn_enable,
            benchmark=self.cudnn_benchmark,
            deterministic=self.cudnn_deterministic,
        ):
            return torch.log_softmax(x, dim=self.dim)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed