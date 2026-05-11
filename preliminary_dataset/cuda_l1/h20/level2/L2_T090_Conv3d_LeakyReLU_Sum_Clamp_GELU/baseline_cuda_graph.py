import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D convolution, applies LeakyReLU, sums with a tensor, clamps, and applies GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, sum_tensor_shape):
        super(Model, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        self.sum_tensor = nn.Parameter(torch.randn(sum_tensor_shape))
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # Fallback to eager execution for non-CUDA tensors
        if not x.is_cuda:
            x = self.conv(x)
            x = torch.nn.functional.leaky_relu(x, negative_slope=0.2)
            x = x + self.sum_tensor
            x = torch.clamp(x, min=-1.0, max=1.0)
            x = torch.nn.functional.gelu(x)
            return x

        # First run: capture the graph
        if self.graph is None:
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = x.clone()
            
            with torch.cuda.graph(self.graph):
                # The actual model operations are captured here
                y = self.conv(self.static_input)
                y = torch.nn.functional.leaky_relu(y, negative_slope=0.2)
                y = y + self.sum_tensor
                y = torch.clamp(y, min=-1.0, max=1.0)
                self.static_output = torch.nn.functional.gelu(y)

        # Copy input data to the static tensor and replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = 3
sum_tensor_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, sum_tensor_shape]