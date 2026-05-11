import torch
import torch.nn as nn

class Model(nn.Module):
    """
    This model performs a convolution, average pooling, applies sigmoid, and sums the result.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.avg_pool = nn.AvgPool2d(pool_kernel_size)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first forward pass, the graph is not yet captured.
        if self.graph is None:
            # Create a persistent static input tensor. It's crucial this is an 
            # attribute of the model to avoid side-effects on the caller's tensor.
            self.static_input = torch.empty_like(x)
            
            # --- Graph Capture ---
            # All operations inside the 'with' block are recorded into the graph.
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                # The model's forward pass is defined using the static tensors.
                y = self.conv(self.static_input)
                y = self.avg_pool(y)
                y = torch.sigmoid(y)
                self.static_output = torch.sum(y, dim=[1,2,3])
            
            # Save the captured graph for future replays.
            self.graph = g

        # For every run (including the first), copy the input data into our 
        # static tensor and replay the captured graph.
        self.static_input.copy_(x)
        self.graph.replay()
        
        return self.static_output

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]