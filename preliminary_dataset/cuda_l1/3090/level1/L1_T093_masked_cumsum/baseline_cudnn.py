import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim, cudnn_enabled=True, cudnn_benchmark=False, cudnn_deterministic=False):
        super(Model, self).__init__()
        self.dim = dim
        self.cudnn_enabled = cudnn_enabled
        self.cudnn_benchmark = cudnn_benchmark
        self.cudnn_deterministic = cudnn_deterministic

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        with torch.backends.cudnn.flags(enabled=self.cudnn_enabled, benchmark=self.cudnn_benchmark, deterministic=self.cudnn_deterministic):
            return torch.cumsum(x * mask, dim=self.dim)

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    x = torch.randn(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()  # Random boolean mask
    return [x, mask]

def get_init_inputs():
    return [dim]