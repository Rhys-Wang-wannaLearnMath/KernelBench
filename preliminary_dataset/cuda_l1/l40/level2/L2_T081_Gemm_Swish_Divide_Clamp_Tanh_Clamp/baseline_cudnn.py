import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a gemm, swish, divide, clamp, tanh, and clamp operations.
    """
    def __init__(self, in_features, out_features, bias=True, cudnn_benchmark=False, cudnn_deterministic=False, cudnn_allow_tf32=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.cudnn_benchmark = cudnn_benchmark
        self.cudnn_deterministic = cudnn_deterministic
        self.cudnn_allow_tf32 = cudnn_allow_tf32

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        with torch.backends.cudnn.flags(benchmark=self.cudnn_benchmark, deterministic=self.cudnn_deterministic, allow_tf32=self.cudnn_allow_tf32):
            x = self.gemm(x)
            x = x * torch.sigmoid(x)  # Swish activation
            x = x / 2.0
            x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
            x = torch.tanh(x)  # Tanh activation
            x = torch.clamp(x, min=-1.0, max=1.0)  # Clamp between -1 and 1
            return x

batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]