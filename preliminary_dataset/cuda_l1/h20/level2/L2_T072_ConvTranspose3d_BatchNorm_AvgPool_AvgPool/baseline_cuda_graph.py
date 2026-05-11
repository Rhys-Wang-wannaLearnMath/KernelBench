import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a 3D transposed convolution, followed by batch normalization, 
    two average pooling layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)

        # Placeholders for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first forward pass, capture the graph.
        if self.graph is None:
            # Create a CUDA graph object.
            self.graph = torch.cuda.CUDAGraph()
            
            # Create a static input tensor. The graph will be captured with this.
            self.static_input = x.clone()

            # Begin graph capture on the current stream.
            with torch.cuda.graph(self.graph):
                # Run the original forward pass logic to trace the operations.
                y = self.conv_transpose(self.static_input)
                y = self.batch_norm(y)
                y = self.avg_pool1(y)
                y = self.avg_pool2(y)
                # The final tensor in the captured region is our static output.
                self.static_output = y
        
        # For every run (including the first), copy the new input data into
        # the memory region of the static input tensor.
        self.static_input.copy_(x)
        
        # Replay the captured graph.
        self.graph.replay()
        
        # Return a clone of the static output tensor.
        return self.static_output.clone()


batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]