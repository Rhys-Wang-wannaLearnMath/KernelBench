import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a 3D convolution, applies minimum operation along a specific dimension, 
    and then applies softmax.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dim):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.dim = dim
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, D, H, W)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, H, W)
        """
        # CUDA graphs are typically used for inference.
        # Fallback to eager execution if in training mode.
        if self.training:
            y = self.conv(x)
            y = torch.min(y, dim=self.dim)[0]
            y = torch.softmax(y, dim=1)
            return y

        # Inference mode with CUDA graph
        if self.graph is None:
            # First inference run: capture the graph.
            # The shape of the first input tensor determines the static shape for the graph.
            self.static_input = x.clone()

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # Define the graph by running the forward pass with the static input.
                y = self.conv(self.static_input)
                y = torch.min(y, dim=self.dim)[0]
                self.static_output = torch.softmax(y, dim=1)
            
            # For the first run, replay the captured graph to compute the output.
            self.graph.replay()
            return self.static_output
        else:
            # Subsequent inference runs: update the input and replay the graph.
            self.static_input.copy_(x)
            self.graph.replay()
            return self.static_output

batch_size = 128
in_channels = 3
out_channels = 16
D, H, W = 16, 32, 32
kernel_size = 3
dim = 2  # Dimension along which to apply minimum operation (e.g., depth)

def get_inputs():
    return [torch.randn(batch_size, in_channels, D, H, W)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, dim]