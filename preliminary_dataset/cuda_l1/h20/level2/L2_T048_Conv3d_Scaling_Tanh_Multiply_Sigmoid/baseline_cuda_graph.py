import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, scales the output, applies tanh, multiplies by a scaling factor, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scaling_factor, bias_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.scaling_factor = nn.Parameter(torch.randn(bias_shape))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None
        self.graph_captured = False

    def forward(self, x):
        if self.training or not self.graph_captured:
            x = self.conv(x)
            x = x * self.scaling_factor 
            x = torch.tanh(x)
            x = x * self.bias
            x = torch.sigmoid(x)
            return x
        
        if self.cuda_graph is None:
            # Capture the graph
            self.static_input = x.clone()
            self.static_output = torch.empty_like(self._compute_output_shape(x))
            
            torch.cuda.synchronize()
            self.cuda_graph = torch.cuda.CUDAGraph()
            
            with torch.cuda.graph(self.cuda_graph):
                temp = self.conv(self.static_input)
                temp = temp * self.scaling_factor 
                temp = torch.tanh(temp)
                temp = temp * self.bias
                self.static_output = torch.sigmoid(temp)
        
        # Copy input data and replay graph
        self.static_input.copy_(x)
        self.cuda_graph.replay()
        return self.static_output.clone()
    
    def _compute_output_shape(self, x):
        with torch.no_grad():
            temp = self.conv(x)
            temp = temp * self.scaling_factor 
            temp = torch.tanh(temp)
            temp = temp * self.bias
            temp = torch.sigmoid(temp)
            return temp

    def enable_cuda_graph(self):
        self.graph_captured = True
        
    def disable_cuda_graph(self):
        self.graph_captured = False
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
scaling_factor = 2
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scaling_factor, bias_shape]