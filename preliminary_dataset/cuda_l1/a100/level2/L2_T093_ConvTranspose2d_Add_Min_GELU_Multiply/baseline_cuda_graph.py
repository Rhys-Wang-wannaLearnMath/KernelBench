import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, adds a value, takes the minimum, applies GELU, and multiplies by a value.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # Create static tensors for graph capture
            self.static_input = torch.empty_like(x)
            self.static_input.copy_(x)
            
            # Create tensor on the same device for min operation
            zero_tensor = torch.tensor(0.0, device=x.device)
            
            # Synchronize before graph capture
            torch.cuda.synchronize()
            
            # Capture the graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                x_graph = self.conv_transpose(self.static_input)
                x_graph = x_graph + self.add_value
                x_graph = torch.min(x_graph, zero_tensor)
                x_graph = torch.nn.functional.gelu(x_graph)
                self.static_output = x_graph * self.multiply_value
            
        # Copy input to static tensor and replay graph
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output

batch_size = 128
in_channels = 32
out_channels = 16
height, width = 32, 32
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]