import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a GELU activation.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.cudnn_enabled = True
        self.cudnn_benchmark = True
        self.cudnn_deterministic = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies GELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with GELU applied, same shape as input.
        """
        with torch.backends.cudnn.flags(
            enabled=self.cudnn_enabled,
            benchmark=self.cudnn_benchmark,
            deterministic=self.cudnn_deterministic
        ):
            return torch.nn.functional.gelu(x)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed