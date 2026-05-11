import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Mish, and applies Mish again.
    """
    def __init__(self, in_features, out_features, cudnn_enabled=True, cudnn_benchmark=False, cudnn_deterministic=False, cudnn_allow_tf32=True):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.cudnn_flags = {
            "enabled": cudnn_enabled,
            "benchmark": cudnn_benchmark,
            "deterministic": cudnn_deterministic,
            "allow_tf32": cudnn_allow_tf32,
        }

    def forward(self, x):
        with torch.backends.cudnn.flags(**self.cudnn_flags):
            x = self.linear(x)
            x = torch.nn.functional.mish(x)
            x = torch.nn.functional.mish(x)
            return x

batch_size = 128
in_features = 10
out_features = 20

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]