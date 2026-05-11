import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A 3D convolutional transpose layer followed by Batch Normalization and subtraction.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        self.batch_norm = nn.BatchNorm3d(out_channels)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
        # A non-default stream is required for graph capture
        self.stream = torch.cuda.Stream()

    def forward(self, x):
        # If the graph has not been captured yet, capture it.
        if self.graph is None:
            # Create static tensors for the graph.
            # Their shapes are determined by the first input.
            self.static_input = x.clone()

            # Create the graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Capture the graph on a non-default stream.
            with torch.cuda.stream(self.stream):
                self.graph.capture_begin()
                
                # The actual model operations are captured using static tensors.
                y = self.conv_transpose(self.static_input)
                y = self.batch_norm(y)
                self.static_output = y - torch.mean(y, dim=(2, 3, 4), keepdim=True)
                
                self.graph.capture_end()
            
            # Ensure the capture is complete before we proceed.
            torch.cuda.current_stream().wait_stream(self.stream)

        # On every run (including the first one, after capture), copy the new input
        # into the static input tensor and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()
        
        # Return a clone of the static output.
        return self.static_output.clone()

batch_size = 16
in_channels = 16
out_channels = 32
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]