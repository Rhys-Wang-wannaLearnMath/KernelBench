import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs a transposed 1D convolution operation with square input and asymmetric kernel, optionally with dilation.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(Model, self).__init__()
        self.conv1d_transpose = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, bias=bias)
        self.graph = None
        self.static_input = None
        self.static_output = None
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the transposed 1D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, length).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, length_out).
        """
        if self.graph is None:
            # On the first forward pass, we capture the model's operations into a CUDA graph.
            # We clone the input tensor to create a static tensor for the graph.
            self.static_input = x.clone()
            
            # Create a new CUDA graph.
            self.graph = torch.cuda.CUDAGraph()
            
            # Enter the graph capture context.
            with torch.cuda.graph(self.graph):
                # Run the model with the static input to define the graph.
                # The resulting tensor becomes the static output for the graph.
                self.static_output = self.conv1d_transpose(self.static_input)
        
        # For every forward pass (including the first), update the static input tensor's data.
        self.static_input.copy_(x)
        
        # Replay the captured graph. This executes the operations efficiently.
        self.graph.replay()
        
        # Return a clone of the static output. This is crucial to ensure that each
        # call to forward returns a new tensor, matching the behavior of the original
        # model and preventing value mismatches caused by output tensor aliasing.
        return self.static_output.clone()

# Test code
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = 5
length = 256
stride = 1
padding = 0
dilation = 3

def get_inputs():
    x = torch.randn(batch_size, in_channels, length)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]