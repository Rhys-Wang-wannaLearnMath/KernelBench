import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a 3D transposed convolution, followed by a sum, layer normalization, average pooling, and GELU activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding)
        self.sum_weight = nn.Parameter(torch.tensor(sum_weight))
        self.norm = nn.LayerNorm(norm_shape)
        self.avg_pool = nn.AvgPool3d(kernel_size=pool_kernel_size)
        self.gelu = nn.GELU()
        
        # Attributes for CUDA graph
        self.stream = torch.cuda.Stream()
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first forward pass, capture the model's operations in a graph.
        if self.graph is None:
            # The static input tensor must have the same shape/type as the real input.
            self.static_input = x
            
            # Capture the graph on a non-default stream.
            with torch.cuda.stream(self.stream):
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    # Run the forward pass to capture the operations.
                    # The output of this sequence becomes the graph's static output tensor.
                    y = self.conv_transpose(self.static_input)
                    y = y + self.sum_weight
                    y = self.norm(y)
                    y = self.avg_pool(y)
                    y = self.gelu(y)
                    self.static_output = y
            
            # Ensure the capture on the side stream is complete before proceeding on the default stream.
            torch.cuda.current_stream().wait_stream(self.stream)

        # For every run (including the first), copy the new input and replay the graph.
        # These operations are on the default stream, ensuring proper ordering.
        self.static_input.copy_(x)
        self.graph.replay()
        
        return self.static_output

batch_size = 128
in_channels = 32
out_channels = 64
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
stride = (2, 2, 2)
padding = (1, 1, 1)
output_padding = (1, 1, 1)
sum_weight = 1.0
norm_shape = (out_channels,)
pool_kernel_size = (2, 2, 2)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, sum_weight, norm_shape, pool_kernel_size]