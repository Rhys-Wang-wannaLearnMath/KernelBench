import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs Layer Normalization.
    """
    def __init__(self,
                 normalized_shape: tuple,
                 cudnn_benchmark: bool = False,
                 cudnn_deterministic: bool = False,
                 cudnn_allow_tf32: bool = True):
        """
        Initializes the LayerNorm layer.

        Args:
            normalized_shape (tuple): Shape of the input tensor to be normalized.
            cudnn_benchmark (bool): If True, causes cuDNN to benchmark multiple algorithms and select the fastest.
            cudnn_deterministic (bool): If True, causes cuDNN to use deterministic algorithms.
            cudnn_allow_tf32 (bool): If True, allows cuDNN to use the TF32 data type for internal computations.
        """
        super(Model, self).__init__()
        self.ln = nn.LayerNorm(normalized_shape=normalized_shape)
        self.cudnn_benchmark = cudnn_benchmark
        self.cudnn_deterministic = cudnn_deterministic
        self.cudnn_allow_tf32 = cudnn_allow_tf32

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Layer Normalization to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (*, normalized_shape).

        Returns:
            torch.Tensor: Output tensor with Layer Normalization applied, same shape as input.
        """
        with torch.backends.cudnn.flags(benchmark=self.cudnn_benchmark, deterministic=self.cudnn_deterministic, allow_tf32=self.cudnn_allow_tf32):
            return self.ln(x)

batch_size = 16
features = 64
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, features, dim1, dim2)
    return [x]

def get_init_inputs():
    return [(features, dim1, dim2)]