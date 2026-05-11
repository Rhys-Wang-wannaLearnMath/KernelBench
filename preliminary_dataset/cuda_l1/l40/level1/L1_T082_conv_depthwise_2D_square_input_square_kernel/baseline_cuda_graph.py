import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a depthwise 2D convolution operation with square input and square kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, bias: bool = False):
        super(Model, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, groups=in_channels, bias=bias)
        
        # CUDA graph attributes
        self.graph = None
        self.static_input = None
        self.static_output = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        if self.graph is None:
            # First forward pass:
            # 1. Perform a regular forward pass to get the correct output. This ensures
            #    that the first call returns a value that is numerically identical to the
            #    non-graphed version, avoiding potential minor discrepancies from the
            #    eager execution during graph capture.
            out = self.conv2d(x)

            # 2. Capture the graph for subsequent runs.
            self.static_input = x.clone()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.static_output = self.conv2d(self.static_input)
            self.graph = g
            
            return out
        else:
            # Subsequent forward passes:
            # 1. Copy the new input data into the static buffer used by the graph.
            self.static_input.copy_(x)
            
            # 2. Replay the captured graph to perform the computation.
            self.graph.replay()
            
            # 3. Return a clone of the output. Cloning is essential to return a new
            #    tensor to the user, preventing the graph's static output buffer from
            #    being inadvertently modified by downstream operations.
            return self.static_output.clone()

# Test code
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding]