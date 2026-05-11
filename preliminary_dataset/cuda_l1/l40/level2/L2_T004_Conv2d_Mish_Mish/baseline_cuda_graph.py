import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, applies Mish, and another Mish.
    """
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        # Initialize attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # On the first pass, capture the model's operations in a CUDA graph.
            self.graph = torch.cuda.CUDAGraph()
            # Create a static tensor to hold the input. This tensor's memory will be
            # reused across all forward passes.
            self.static_input = x.clone()

            with torch.cuda.graph(self.graph):
                # Run the forward pass within the graph capture context.
                # The operations are recorded rather than executed immediately.
                y = self.conv(self.static_input)
                y = torch.nn.functional.mish(y)
                # The output of the captured region must also be a static tensor.
                self.static_output = torch.nn.functional.mish(y)

        # For every forward pass (including the first), copy the current input data
        # into the static input tensor.
        self.static_input.copy_(x)

        # Replay the captured graph. This is much faster than executing the
        # operations individually.
        self.graph.replay()

        # Return a clone of the static output. Cloning is necessary to ensure
        # the graph's internal tensors are not modified by subsequent operations.
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]