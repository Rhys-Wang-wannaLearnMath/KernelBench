import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A 3D convolutional layer followed by multiplication, instance normalization, clamping, multiplication, and a max operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.multiplier = nn.Parameter(torch.randn(multiplier_shape))
        self.instance_norm = nn.InstanceNorm3d(out_channels)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Attributes for CUDA graph functionality
        self.graph = None
        self.static_input = None
        self.static_output = None
        # A non-default stream is required for graph capture
        self.stream = torch.cuda.Stream()

    def forward(self, x):
        if self.graph is None:
            # On the first forward pass, we capture the graph.
            # Graph capture must be done on a non-default stream.
            self.stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.stream):
                # Create a static input tensor. It's important to clone as the input 'x' can change.
                self.static_input = x.clone()

                # Perform a "warmup" run. This initializes stateful layers (like instance_norm)
                # and gives us a correctly-shaped output tensor to use as a static buffer.
                y = self.conv(self.static_input)
                y = y * self.multiplier
                y = self.instance_norm(y)
                y = torch.clamp(y, self.clamp_min, self.clamp_max)
                y = y * self.multiplier
                y = torch.max(y, dim=1)[0]
                self.static_output = y

                # Now, capture the graph using the static tensors.
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    # Rerun the forward pass logic. These operations are recorded into the graph.
                    graph_y = self.conv(self.static_input)
                    graph_y = graph_y * self.multiplier
                    graph_y = self.instance_norm(graph_y)
                    graph_y = torch.clamp(graph_y, self.clamp_min, self.clamp_max)
                    graph_y = graph_y * self.multiplier
                    graph_y = torch.max(graph_y, dim=1)[0]
                    # The graph's output must be copied into our persistent static output buffer.
                    self.static_output.copy_(graph_y)
            
            # Synchronize the default stream to wait for the capture to complete on our stream.
            torch.cuda.current_stream().wait_stream(self.stream)
            
            # The output from the warmup run is the correct output for this first pass.
            return self.static_output
        else:
            # For subsequent passes, copy new data to the static input tensor and replay the graph.
            self.static_input.copy_(x)
            self.graph.replay()
            return self.static_output

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
multiplier_shape = (out_channels, 1, 1, 1)
clamp_min = -1.0
clamp_max = 1.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, multiplier_shape, clamp_min, clamp_max]