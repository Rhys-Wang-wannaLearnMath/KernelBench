import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, applies softmax, adds a bias term, scales the result, and applies sigmoid.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape)) 
        self.scaling_factor = scaling_factor
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # On the first forward pass, capture the graph.
            self.static_input = torch.zeros_like(x)
            
            # --- Begin graph capture ---
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # The model's forward pass logic, using the static input tensor.
                # The resulting output tensor is also stored as a static attribute.
                y = self.conv_transpose(self.static_input)
                y = torch.softmax(y, dim=1)
                y = y + self.bias
                y = y * self.scaling_factor
                self.static_output = torch.sigmoid(y)
            # --- End graph capture ---

        # For every run (including the first), copy the current input's data
        # to the static input tensor and replay the captured graph.
        self.static_input.copy_(x)
        self.graph.replay()
        
        # Return the static output tensor, which has been updated by the graph replay.
        return self.static_output

batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape, scaling_factor]