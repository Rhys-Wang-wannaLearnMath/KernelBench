import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj):
        """
        :param in_channels: Number of input channels
        :param out_1x1: Number of output channels for the 1x1 convolution
        :param reduce_3x3: Number of output channels for the 1x1 reduction before 3x3 convolution
        :param out_3x3: Number of output channels for the 3x3 convolution
        :param reduce_5x5: Number of output channels for the 1x1 reduction before 5x5 convolution
        :param out_5x5: Number of output channels for the 5x5 convolution
        :param pool_proj: Number of output channels for the pooling projection
        """
        super(ModelNew, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3_reduce = nn.Conv2d(in_channels, reduce_3x3, kernel_size=1)
        self.branch3x3 = nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        
        # 5x5 convolution branch
        self.branch5x5_reduce = nn.Conv2d(in_channels, reduce_5x5, kernel_size=1)
        self.branch5x5 = nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        
        # Max pooling branch
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_proj = nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        
        # Store output channel dimensions for efficient slicing
        self.out_1x1 = out_1x1
        self.out_3x3 = out_3x3
        self.out_5x5 = out_5x5
        self.pool_proj = pool_proj
        self.total_channels = out_1x1 + out_3x3 + out_5x5 + pool_proj

    def _compute_branch(self, x, branch_id):
        """Compute a specific branch with tensor caching removed"""
        if branch_id == 1:
            return self.branch1x1(x)
        elif branch_id == 2:
            branch3x3_reduce_out = self.branch3x3_reduce(x)
            return self.branch3x3(branch3x3_reduce_out)
        elif branch_id == 3:
            branch5x5_reduce_out = self.branch5x5_reduce(x)
            return self.branch5x5(branch5x5_reduce_out)
        elif branch_id == 4:
            branch_pool_out = self.branch_pool(x)
            return self.branch_pool_proj(branch_pool_out)
        return None
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        # Compute the reduction operations
        branch3x3_reduce_out = self.branch3x3_reduce(x)
        branch5x5_reduce_out = self.branch5x5_reduce(x)
        branch_pool_out = self.branch_pool(x)
        
        # Compute branches
        branch1x1_out = self.branch1x1(x)
        branch3x3_out = self.branch3x3(branch3x3_reduce_out)
        branch5x5_out = self.branch5x5(branch5x5_reduce_out)
        branch_pool_proj_out = self.branch_pool_proj(branch_pool_out)
        
        # Use pre-allocation for output tensor to avoid concatenation overhead
        batch_size, _, height, width = x.shape
        output = torch.empty(batch_size, self.total_channels, height, width, 
                             device=x.device, dtype=x.dtype)
        
        # Copy each branch output to the corresponding slice of the output tensor
        output[:, :self.out_1x1] = branch1x1_out
        output[:, self.out_1x1:self.out_1x1 + self.out_3x3] = branch3x3_out
        start_idx = self.out_1x1 + self.out_3x3
        output[:, start_idx:start_idx + self.out_5x5] = branch5x5_out
        start_idx = self.out_1x1 + self.out_3x3 + self.out_5x5
        output[:, start_idx:] = branch_pool_proj_out
        
        return output

# Test code
in_channels = 480
out_1x1 = 192
reduce_3x3 = 96
out_3x3 = 208
reduce_5x5 = 16
out_5x5 = 48
pool_proj = 64
batch_size = 10
height = 224
width = 224

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_1x1, reduce_3x3, out_3x3, reduce_5x5, out_5x5, pool_proj]