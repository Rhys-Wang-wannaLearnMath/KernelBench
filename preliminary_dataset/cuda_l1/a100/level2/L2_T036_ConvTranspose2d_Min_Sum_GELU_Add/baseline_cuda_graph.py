import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a convolution transpose, minimum operation, sum operation, GELU activation and addition.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape):
        super(Model, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding)
        self.bias = nn.Parameter(torch.randn(bias_shape))

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # First run: capture the graph.
            # We need a static input tensor with a fixed memory address.
            self.static_input = x.clone()
            
            # Instantiate the graph and start capturing.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # The model's forward logic is recorded using the static input.
                y = self.conv_transpose(self.static_input)
                y = torch.min(y, dim=1, keepdim=True)[0]  # Minimum operation along channel dimension
                y = torch.sum(y, dim=2, keepdim=True)  # Sum operation along height dimension
                y = torch.nn.functional.gelu(y)  # GELU activation
                y = y + self.bias
                # The output tensor's memory location is also captured.
                self.static_output = y
            
        # For all runs (including the first, after capture), copy the new input data to the static buffer.
        self.static_input.copy_(x)
        
        # Replay the captured graph. This executes the recorded kernels
        # with the data in self.static_input and writes the result to self.static_output.
        self.graph.replay()

        # Return a clone of the static output to prevent the caller's tensor
        # from being mutated by the next graph replay.
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
stride = 2
padding = 1
output_padding = 1
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, output_padding, bias_shape]