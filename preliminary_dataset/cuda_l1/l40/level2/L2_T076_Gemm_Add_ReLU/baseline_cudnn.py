import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape, cudnn_enabled=True, cudnn_benchmark=False):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.cudnn_enabled = cudnn_enabled
        self.cudnn_benchmark = cudnn_benchmark

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        with torch.backends.cudnn.flags(enabled=self.cudnn_enabled, benchmark=self.cudnn_benchmark):
            x = self.gemm(x)
            x = x + self.bias
            x = torch.relu(x)
            return x

batch_size = 128
in_features = 1024
out_features = 512
bias_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bias_shape]