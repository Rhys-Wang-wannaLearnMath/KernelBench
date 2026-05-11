import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a GEMM, GroupNorm, Swish, Multiply, and Swish operations.
    """
    def __init__(self, in_features, out_features, num_groups, multiply_weight_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.group_norm = nn.GroupNorm(num_groups, out_features)
        self.multiply_weight = nn.Parameter(torch.randn(multiply_weight_shape))
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first run, capture the graph
        if self.graph is None:
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = x.clone()
            with torch.cuda.graph(self.graph):
                static_y = self.gemm(self.static_input)
                static_y = self.group_norm(static_y)
                static_y = static_y * torch.sigmoid(static_y)
                static_y = static_y * self.multiply_weight
                static_y = static_y * torch.sigmoid(static_y)
                self.static_output = static_y

        # Copy the current input to the static buffer
        self.static_input.copy_(x)
        # Replay the graph
        self.graph.replay()
        # Return a clone of the output
        return self.static_output.clone()

batch_size = 128
in_features = 512
out_features = 1024
num_groups = 16
multiply_weight_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, multiply_weight_shape]