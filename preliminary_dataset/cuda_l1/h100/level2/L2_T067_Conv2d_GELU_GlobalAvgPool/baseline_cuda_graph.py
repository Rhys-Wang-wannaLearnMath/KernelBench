import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies GELU, and then performs global average pooling.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # CUDA graph attributes
        self.graph = None
        self.stream = torch.cuda.Stream()
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels)
        """
        # On the first iteration, capture the CUDA graph on a non-default stream.
        if self.graph is None:
            # Use the non-default stream for capture.
            with torch.cuda.stream(self.stream):
                # Create a static input tensor that will be used for all subsequent runs.
                self.static_input = x.clone()

                # Create the graph object and capture the model's operations.
                self.graph = torch.cuda.CUDAGraph()
                self.graph.capture_begin()

                y = self.conv(self.static_input)
                y = torch.nn.functional.gelu(y)
                y = torch.nn.functional.adaptive_avg_pool2d(y, 1)
                y = y.squeeze(-1).squeeze(-1)
                self.static_output = y

                self.graph.capture_end()

            # Wait for the capture to finish on the side stream before proceeding.
            torch.cuda.current_stream().wait_stream(self.stream)

        # For every iteration, copy the new input data to the static buffer.
        self.static_input.copy_(x)

        # Replay the captured graph to perform the computation.
        self.graph.replay()

        # Return the static output tensor.
        return self.static_output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]