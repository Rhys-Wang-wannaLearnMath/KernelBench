import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a 2D transposed convolution operation with asymmetric input and square kernel, supporting dilation, padding, and stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square, e.g., 3 for a 3x3 kernel).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv_transpose2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
        # A non-default stream is required for graph capture
        self.stream = torch.cuda.Stream()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height_in, width_in). 

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        if self.graph is None:
            # On the first run, capture the graph.
            # Create static placeholders that will be used for all subsequent runs.
            self.static_input = torch.empty_like(x)
            
            # Initialize the graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Capture the model's operations on a non-default stream.
            with torch.cuda.stream(self.stream):
                self.graph.capture_begin()
                self.static_output = self.conv_transpose2d(self.static_input)
                self.graph.capture_end()

        # For every run, copy the input data to the static placeholder.
        self.static_input.copy_(x)
        # Replay the captured graph.
        self.graph.replay()
        
        # The result is in the static output tensor after replay.
        return self.static_output


# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
height_in = 64
width_in = 128
stride = 5
padding = 1
dilation = 2

def get_inputs():
    x = torch.randn(batch_size, in_channels, height_in, width_in)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]