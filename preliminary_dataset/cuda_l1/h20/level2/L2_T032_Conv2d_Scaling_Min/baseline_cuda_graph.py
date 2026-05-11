import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, scales the output, and then applies a minimum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.scale_factor = scale_factor

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
        """
        if self.graph is None:
            # On the first run, we capture the graph.
            self.static_input = torch.empty_like(x)
            
            # Create a new graph object.
            self.graph = torch.cuda.CUDAGraph()
            
            # Enter capture mode.
            with torch.cuda.graph(self.graph):
                # Perform the model operations using the static input tensor.
                # The result is assigned to the static output tensor.
                out = self.conv(self.static_input)
                out = out * self.scale_factor
                self.static_output = torch.min(out, dim=1, keepdim=True)[0]

        # For every run (including the first), copy the current input to the static input tensor,
        # replay the graph, and return a clone of the static output.
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
scale_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, scale_factor]