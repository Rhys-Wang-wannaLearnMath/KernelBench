import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    """
    Simple model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0, cudnn_enabled: bool = True, cudnn_benchmark: bool = False, cudnn_deterministic: bool = False):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
            cudnn_enabled (bool, optional): Enables or disables cudnn. Defaults to True.
            cudnn_benchmark (bool, optional): Enables or disables cudnn benchmarking. Defaults to False.
            cudnn_deterministic (bool, optional): Enables or disables cudnn deterministic mode. Defaults to False.
        """
        super(Model, self).__init__()
        self.alpha = alpha
        
        # Set CuDNN backend flags
        torch.backends.cudnn.enabled = cudnn_enabled
        torch.backends.cudnn.benchmark = cudnn_benchmark
        torch.backends.cudnn.deterministic = cudnn_deterministic

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        return F.elu(x, alpha=self.alpha)

batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization