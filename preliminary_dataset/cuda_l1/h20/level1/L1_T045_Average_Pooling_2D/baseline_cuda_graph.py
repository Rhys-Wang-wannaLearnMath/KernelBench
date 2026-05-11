import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs 2D Average Pooling.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0):
        """
        Initializes the Average Pooling layer.

        Args:
            kernel_size (int): Size of the pooling window.
            stride (int, optional): Stride of the pooling operation. Defaults to None (same as kernel_size).
            padding (int, optional): Padding applied to the input tensor. Defaults to 0.
        """
        super(Model, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=kernel_size, stride=stride, padding=padding)
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies 2D Average Pooling to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor with Average Pooling applied.
        """
        # The first forward pass will capture the graph.
        # Subsequent calls will replay the graph.
        if self.graph is None:
            # Create static tensors for inputs and outputs.
            # This is necessary because the graph is defined for specific tensor memory addresses.
            self.static_input = x.clone()

            # Create the CUDA graph.
            self.graph = torch.cuda.CUDAGraph()

            # Enter graph capture mode.
            with torch.cuda.graph(self.graph):
                # Run the model's operations. The output tensor's memory will be captured.
                self.static_output = self.avg_pool(self.static_input)
        
        # Copy the current input data to the static input tensor's memory.
        self.static_input.copy_(x)
        
        # Replay the captured graph.
        self.graph.replay()
        
        # Return the output from the static output tensor.
        return self.static_output

batch_size = 16
channels = 64
height = 256
width = 256
kernel_size = 3

def get_inputs():
    x = torch.randn(batch_size, channels, height, width)
    return [x]

def get_init_inputs():
    return [kernel_size]