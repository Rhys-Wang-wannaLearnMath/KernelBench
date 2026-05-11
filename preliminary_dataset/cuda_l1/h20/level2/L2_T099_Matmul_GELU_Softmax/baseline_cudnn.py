import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features, cudnn_benchmark=False, cudnn_deterministic=False):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.cudnn_benchmark = cudnn_benchmark
        self.cudnn_deterministic = cudnn_deterministic

    def forward(self, x):
        with torch.backends.cudnn.flags(benchmark=self.cudnn_benchmark, deterministic=self.cudnn_deterministic):
            x = self.linear(x)
            x = torch.nn.functional.gelu(x)
            x = torch.nn.functional.softmax(x, dim=1)
            return x

batch_size = 128
in_features = 100
out_features = 10

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]