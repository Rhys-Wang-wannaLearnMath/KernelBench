import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 3D convolution operation with an asymmetric input and a square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel (kernel_size x kernel_size).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv3d = nn.Conv3d(in_channels, out_channels, (kernel_size, kernel_size, 1), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        self.graph = None
        self.static_input = None
        self.static_output = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out, depth_out).
        """
        if self.graph is None:
            # First run: capture the graph.
            # The graph requires static memory addresses for inputs and outputs.
            self.static_input = x.clone()
            self.graph = torch.cuda.CUDAGraph()

            with torch.cuda.graph(self.graph):
                # The operations are run once during capture to define the graph
                # and determine the output tensor's properties.
                self.static_output = self.conv3d(self.static_input)
            
            # The result for the first input is now in self.static_output.
            # We proceed to the common return path.

        # For subsequent runs, copy the new input data and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()

        # Return a clone of the static output. This is crucial to prevent the caller
        # from holding a reference to the internal graph buffer, which would be
        # overwritten in subsequent forward passes.
        return self.static_output.clone()

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256
depth = 10

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width, depth)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization