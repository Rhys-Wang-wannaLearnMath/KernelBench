import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Mish activation, and then applies Tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def _core_forward(self, x):
        """The actual model logic, to be captured by the graph."""
        x = self.conv(x)
        x = torch.nn.functional.mish(x)
        x = torch.tanh(x)
        return x

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, D', H', W').
        """
        if self.graph is None:
            # First run: capture the graph.
            # The graph is captured on the default stream.
            self.static_input = torch.empty_like(x)
            
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.static_output = self._core_forward(self.static_input)
            self.graph = g

        # On every run (including the first), copy the input to the static buffer
        # and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output

batch_size = 16
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]