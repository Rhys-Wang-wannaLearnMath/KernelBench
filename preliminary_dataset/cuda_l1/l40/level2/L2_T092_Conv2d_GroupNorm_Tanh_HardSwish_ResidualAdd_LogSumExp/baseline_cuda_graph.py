import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a convolution, applies Group Normalization, Tanh, HardSwish, 
    Residual Addition, and LogSumExp.
    """
    def __init__(self, in_channels, out_channels, kernel_size, groups, eps=1e-5):
        super(Model, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.group_norm = nn.GroupNorm(groups, out_channels, eps=eps)
        self.tanh = nn.Tanh()
        self.hard_swish = nn.Hardswish()
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # If the graph is not captured yet, capture it on the first run.
        if self.graph is None:
            # Create a static input tensor that will be used for graph capture.
            self.static_input = x.clone()

            # Create a CUDA graph object.
            self.graph = torch.cuda.CUDAGraph()
            
            # Enter graph capture mode.
            with torch.cuda.graph(self.graph):
                # Run the forward pass with the static input to capture the operations.
                # Convolution
                x_conv = self.conv(self.static_input)
                # Group Normalization
                x_norm = self.group_norm(x_conv)
                # Tanh
                x_tanh = self.tanh(x_norm)
                # HardSwish
                x_hard_swish = self.hard_swish(x_tanh)
                # Residual Addition
                x_res = x_conv + x_hard_swish
                # LogSumExp
                x_logsumexp = torch.logsumexp(x_res, dim=1, keepdim=True)
                
                # The final tensor of the graph becomes the static output.
                self.static_output = x_logsumexp

        # For every run (including the first), copy the current input data
        # into the static input tensor and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()
        
        # Return a clone of the static output tensor.
        return self.static_output.clone()

batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
groups = 8

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, groups]