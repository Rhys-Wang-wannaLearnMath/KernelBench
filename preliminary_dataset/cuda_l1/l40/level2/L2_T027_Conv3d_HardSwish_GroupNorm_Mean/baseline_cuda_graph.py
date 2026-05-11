import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a 3D convolution, applies HardSwish, ReLU, Softmax, and then calculates the mean.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=True):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, bias=bias)
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # First call - capture the graph
            self.static_input = x.clone()
            self.static_output = self._forward_impl(self.static_input)
            
            # Capture the graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self._forward_impl(self.static_input)
        
        # Copy input data to static input
        self.static_input.copy_(x)
        
        # Replay the graph
        self.graph.replay()
        
        return self.static_output.clone()
    
    def _forward_impl(self, x):
        x = self.conv(x)
        x = torch.nn.functional.hardswish(x)
        x = torch.relu(x)
        x = torch.softmax(x, dim=1)
        x = torch.mean(x, dim=[2, 3, 4])
        return x

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]