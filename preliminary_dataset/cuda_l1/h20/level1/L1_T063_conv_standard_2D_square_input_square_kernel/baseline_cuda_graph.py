import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a standard 2D convolution operation with a square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the square convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, (kernel_size, kernel_size), stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution using CUDA graphs for CUDATensors.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # On the first forward pass, the graph is not yet captured.
        # We need to capture the graph on the first run.
        if self.graph is None:
            # Create a static input tensor with the same shape and device as the input.
            # This tensor will be used by the CUDA graph.
            self.static_input = torch.empty_like(x)
            
            # Instantiate the CUDA graph
            self.graph = torch.cuda.CUDAGraph()

            # Begin capturing the graph. All operations within this context
            # will be recorded.
            with torch.cuda.graph(self.graph):
                # The forward pass is run once with the static input to define the graph.
                self.static_output = self.conv2d(self.static_input)
        
        # Copy the data from the current input tensor to the static input tensor
        # used by the graph.
        self.static_input.copy_(x)
        
        # Replay the captured graph. This executes the recorded operations
        # with the updated data in self.static_input.
        self.graph.replay()
        
        # Return the output from the graph's static output tensor.
        return self.static_output

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 3
width = 256
height = 256

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization