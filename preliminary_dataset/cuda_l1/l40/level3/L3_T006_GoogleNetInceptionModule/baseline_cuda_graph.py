import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
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
        super(Model, self).__init__()
        
        # 1x1 convolution branch
        self.branch1x1 = nn.Conv2d(in_channels, out_1x1, kernel_size=1)
        
        # 3x3 convolution branch
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_3x3, kernel_size=1),
            nn.Conv2d(reduce_3x3, out_3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 convolution branch
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, reduce_5x5, kernel_size=1),
            nn.Conv2d(reduce_5x5, out_5x5, kernel_size=5, padding=2)
        )
        
        # Max pooling branch
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )
        
        # CUDA Graph attributes
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        if not x.is_cuda:
            # Fallback to eager mode for non-CUDA inputs
            branch1x1 = self.branch1x1(x)
            branch3x3 = self.branch3x3(x)
            branch5x5 = self.branch5x5(x)
            branch_pool = self.branch_pool(x)
            
            outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
            return torch.cat(outputs, 1)

        # First run or shape change: Capture the graph
        if self.graph is None or self.static_input.shape != x.shape:
            # Create static tensors for graph memory.
            self.static_input = torch.empty_like(x)
            
            # Capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                branch1x1 = self.branch1x1(self.static_input)
                branch3x3 = self.branch3x3(self.static_input)
                branch5x5 = self.branch5x5(self.static_input)
                branch_pool = self.branch_pool(self.static_input)
                
                outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
                self.static_output = torch.cat(outputs, 1)

        # Copy input data into the static buffer.
        self.static_input.copy_(x)
        
        # Replay the captured graph.
        self.graph.replay()
        
        # Return a clone of the output to prevent user modification of graph memory.
        return self.static_output.clone()

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