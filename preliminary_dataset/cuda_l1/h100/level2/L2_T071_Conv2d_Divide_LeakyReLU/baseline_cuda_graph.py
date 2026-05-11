import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a convolution, divides by a constant, and applies LeakyReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.divisor = divisor
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # If the graph is not yet captured, it's the first run.
        if self.graph is None:
            # 1. Perform a regular forward pass to get the correct output for this first call.
            #    This is critical for correctness, as the tensor from a capture context
            #    is not the valid result for the initial run.
            y = self.conv(x)
            y = y / self.divisor
            y = torch.nn.functional.leaky_relu(y, negative_slope=0.01)

            # 2. Now, capture the graph for subsequent runs.
            #    We create a static input tensor that will be used for all graph replays.
            self.static_input = x.clone()
            
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                # Trace the forward pass with the static input. The resulting
                # tensor is stored as the static output for the graph.
                static_y = self.conv(self.static_input)
                static_y = static_y / self.divisor
                self.static_output = torch.nn.functional.leaky_relu(static_y, negative_slope=0.01)

            self.graph = g

            # 3. Return the result from the initial, non-graphed run.
            return y
        
        # If the graph has been captured, replay it for performance.
        else:
            # Copy the new input data into the memory of the static input tensor.
            self.static_input.copy_(x)
            
            # Replay the captured graph. This executes the traced operations
            # and writes the result into self.static_output.
            self.graph.replay()
            
            # Return the static output tensor, which now contains the updated result.
            return self.static_output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
divisor = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, divisor]