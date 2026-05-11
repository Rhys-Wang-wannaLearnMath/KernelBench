import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Max reduction over a specific dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): The dimension to reduce over.
        """
        super(Model, self).__init__()
        self.dim = dim
        # cudnn backend flags
        self.cudnn_benchmark = False
        self.cudnn_deterministic = False
        self.cudnn_allow_tf32 = True
        self.cudnn_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        with torch.backends.cudnn.flags(
            benchmark=self.cudnn_benchmark,
            deterministic=self.cudnn_deterministic,
            allow_tf32=self.cudnn_allow_tf32,
            enabled=self.cudnn_enabled
        ):
            return torch.max(x, dim=self.dim)[0]

batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1] # Example, change to desired dimension