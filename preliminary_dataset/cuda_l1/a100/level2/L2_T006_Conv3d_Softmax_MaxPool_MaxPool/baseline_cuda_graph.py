import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies Softmax, and performs two max pooling operations.
    """
    def __init__(self, in_channels, out_channels, kernel_size, pool_kernel_size):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.pool1 = nn.MaxPool3d(pool_kernel_size)
        self.pool2 = nn.MaxPool3d(pool_kernel_size)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, in_channels, depth, height, width)
        Returns:
            Output tensor of shape (batch_size, out_channels, depth', height', width') where depth', height', width' are the dimensions after pooling.
        """
        # On the first forward pass, capture the model's operations in a CUDA graph.
        if self.graph is None:
            # Create a static input tensor. This is necessary because the graph's memory
            # pointers are fixed. We will copy data to this tensor in subsequent calls.
            self.static_input = x.clone()

            # Create a CUDA graph object.
            self.graph = torch.cuda.CUDAGraph()

            # The 'with' block specifies the scope of the graph capture.
            with torch.cuda.graph(self.graph):
                # Run the forward pass using the static input to record the operations.
                # The output of these operations is stored in a static output tensor.
                static_y = self.conv(self.static_input)
                static_y = torch.softmax(static_y, dim=1)
                static_y = self.pool1(static_y)
                self.static_output = self.pool2(static_y)

        # Copy the data from the current input tensor 'x' to the static input tensor.
        self.static_input.copy_(x)
        
        # Replay the captured graph. This executes the recorded operations on the
        # data that was just copied to the static input tensor.
        self.graph.replay()
        
        # Return a clone of the static output. Cloning is important to avoid
        # modifications to the graph's static output buffer from outside.
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
pool_kernel_size = 2

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, pool_kernel_size]