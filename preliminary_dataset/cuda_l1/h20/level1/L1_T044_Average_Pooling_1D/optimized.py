import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized model that performs 1D Average Pooling.
    
    Args:
        kernel_size (int): Size of the pooling window.
        stride (int, optional): Stride of the pooling operation. Defaults to 1.
        padding (int, optional): Padding applied to the input tensor. Defaults to 0.
    """
    def __init__(self, kernel_size, stride=1, padding=0):
        super(ModelNew, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.pool_fn = F.avg_pool1d  # Cache function reference
    
    def forward(self, x):
        """
        Applies optimized 1D Average Pooling to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, input_length).
            
        Returns:
            torch.Tensor: Output tensor with 1D Average Pooling applied.
        """
        # Direct function call to avoid module overhead
        return self.pool_fn(x, self.kernel_size, self.stride, self.padding)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 32
input_length = 128
kernel_size = 4
stride = 2
padding = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, input_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding]