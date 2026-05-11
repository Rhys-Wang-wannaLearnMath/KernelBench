import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, 
    a residual add, a multiplication, and another residual add.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first pass, capture the graph
        if self.graph is None:
            self.static_input = x.clone()
            
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # Original forward logic, but using static tensors
                graph_x = self.conv_transpose(self.static_input)
                graph_original_x = graph_x.clone().detach()
                graph_x = graph_x + self.bias
                graph_x = graph_x + graph_original_x
                graph_x = graph_x * graph_original_x
                self.static_output = graph_x + graph_original_x
        
        # For all passes (including the first), copy input data and replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

batch_size = 16
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]