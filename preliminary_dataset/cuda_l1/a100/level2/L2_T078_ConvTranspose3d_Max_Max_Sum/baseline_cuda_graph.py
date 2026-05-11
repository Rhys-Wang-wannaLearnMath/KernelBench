import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by two max pooling layers and a sum operation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.max_pool1 = nn.MaxPool3d(kernel_size=2)
        self.max_pool2 = nn.MaxPool3d(kernel_size=3)
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
        self.stream = torch.cuda.Stream()

    def forward(self, x):
        # On the first run, capture the CUDA graph on a non-default stream
        if self.graph is None:
            # Create static tensors for capture
            self.static_input = torch.randn_like(x)
            
            # Create a new CUDA graph
            self.graph = torch.cuda.CUDAGraph()

            # Capture the graph within the context of our dedicated stream
            with torch.cuda.stream(self.stream):
                self.graph.capture_begin()
                
                # Run the forward pass to trace the operations
                y = self.conv_transpose(self.static_input)
                y = self.max_pool1(y)
                y = self.max_pool2(y)
                self.static_output = torch.sum(y, dim=1, keepdim=True) 
                
                self.graph.capture_end()

            # It's important to synchronize after capture to ensure it's complete
            torch.cuda.synchronize()

        # For all runs (including the first one after capture),
        # copy the new input data and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()
        
        # Return the static output tensor, which has been updated in-place by the replay
        return self.static_output

batch_size = 16
in_channels = 8
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
stride = 2
padding = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding]