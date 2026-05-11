import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a transposed convolution, followed by max pooling, hardtanh activation, mean operation, and tanh activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.maxpool = nn.MaxPool2d(kernel_size=maxpool_kernel_size, stride=maxpool_stride)
        self.hardtanh = nn.Hardtanh(min_val=hardtanh_min, max_val=hardtanh_max)
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first forward pass, the graph is not yet captured.
        if self.graph is None:
            # Create static tensors for inputs and outputs. These will be reused
            # across subsequent forward passes.
            self.static_input = torch.empty_like(x)

            # Create and capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # Place the original forward pass logic inside the graph capture context.
                # All operations are performed on the static input tensor.
                y = self.conv_transpose(self.static_input)
                y = self.maxpool(y)
                y = self.hardtanh(y)
                y = torch.mean(y, dim=(2, 3), keepdim=True)
                y = torch.tanh(y)
                # The final result is stored in the static output tensor.
                self.static_output = y
        
        # For every forward pass (including the first one), copy the current input
        # data into the graph's static input tensor.
        self.static_input.copy_(x)

        # Replay the captured graph to execute the model's operations.
        self.graph.replay()

        # Return a clone of the static output tensor. Cloning is crucial to prevent
        # downstream operations from corrupting the graph's memory buffer.
        return self.static_output.clone()

batch_size = 128
in_channels = 32
out_channels = 64
height, width = 16, 16
kernel_size = 4
stride = 2
padding = 1
maxpool_kernel_size = 2
maxpool_stride = 2
hardtanh_min = -1
hardtanh_max = 1

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, maxpool_kernel_size, maxpool_stride, hardtanh_min, hardtanh_max]