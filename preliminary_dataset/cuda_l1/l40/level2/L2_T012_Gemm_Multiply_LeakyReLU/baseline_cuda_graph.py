import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a Gemm, multiplies the result, and applies LeakyReLU.
    """
    def __init__(self, in_features, out_features, multiplier, negative_slope):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features)
        self.multiplier = multiplier
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first forward pass, capture the graph.
        if self.graph is None:
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = x.clone()
            
            with torch.cuda.graph(self.graph):
                y = self.gemm(self.static_input)
                y = y * self.multiplier
                self.static_output = self.leaky_relu(y)

        # For all runs, copy the new input and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()
        
        # Return a clone of the static output tensor.
        return self.static_output.clone()

batch_size = 128
in_features = 1024
out_features = 512
multiplier = 2.0
negative_slope = 0.1

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, multiplier, negative_slope]