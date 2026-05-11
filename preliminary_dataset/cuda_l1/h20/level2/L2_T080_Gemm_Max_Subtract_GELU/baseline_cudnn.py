import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a GEMM, followed by a max operation, subtraction, and GELU activation.
    """
    def __init__(self, in_features, out_features, max_dim):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.max_dim = max_dim
        self.cudnn_benchmark = False
        self.cudnn_deterministic = False

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_features)

        Returns:
            Output tensor of shape (batch_size, out_features)
        """
        with torch.backends.cudnn.flags(benchmark=self.cudnn_benchmark, deterministic=self.cudnn_deterministic):
            x = self.gemm(x)
            x = torch.max(x, dim=self.max_dim, keepdim=True).values
            x = x - x.mean(dim=1, keepdim=True)
            x = torch.nn.functional.gelu(x)
            return x

batch_size = 128
in_features = 512
out_features = 1024
max_dim = 1

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, max_dim]