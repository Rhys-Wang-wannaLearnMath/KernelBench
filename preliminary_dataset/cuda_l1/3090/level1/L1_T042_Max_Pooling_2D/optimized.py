import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized model that performs Max Pooling 2D.
    
    Args:
        kernel_size (int): Size of the pooling window.
        stride (int): Stride of the pooling window.
        padding (int): Padding to be applied before pooling.
        dilation (int): Spacing between kernel elements.
    """
    def __init__(self, kernel_size: int, stride: int, padding: int, dilation: int):
        super(ModelNew, self).__init__()
        # Store parameters directly as class attributes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        
        # Pre-compute output dimensions for common input size
        self.out_h = (height + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
        self.out_w = (width + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 2D to the input tensor with minimal overhead.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            
        Returns:
            torch.Tensor: Output tensor after Max Pooling 2D.
        """
        # Direct call to F.max_pool2d with minimal parameter passing
        return F.max_pool2d(
            x, 
            self.kernel_size, 
            self.stride, 
            self.padding, 
            self.dilation, 
            False,  # ceil_mode
            False   # return_indices
        )

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
channels = 32
height = 128
width = 128
kernel_size = 2
stride = 2
padding = 1
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation]