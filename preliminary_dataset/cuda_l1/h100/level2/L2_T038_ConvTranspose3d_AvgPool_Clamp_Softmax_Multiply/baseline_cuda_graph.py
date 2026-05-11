import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, average pooling, clamping, softmax, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.avg_pool = nn.AvgPool3d(pool_kernel_size)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth, height, width).
        """
        if self.graph is None:
            # On the first forward pass, we record the model execution into a graph.
            self.graph = torch.cuda.CUDAGraph()
            # We create static tensors to hold the input and output of the graph.
            self.static_input = x.clone()
            
            with torch.cuda.graph(self.graph):
                # The original forward logic is placed inside the graph capture context.
                static_y = self.conv_transpose(self.static_input)
                static_y = self.avg_pool(static_y)
                static_y = torch.clamp(static_y, self.clamp_min, self.clamp_max)
                static_y = torch.softmax(static_y, dim=1)
                static_y = static_y * 2
                self.static_output = static_y

        # For every run (including the first), copy the current input to the static input tensor...
        self.static_input.copy_(x)
        # ...and replay the graph.
        self.graph.replay()
        # Return a clone of the static output tensor.
        return self.static_output.clone()

batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
pool_kernel_size = 2
clamp_min = 0.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, pool_kernel_size, clamp_min, clamp_max]