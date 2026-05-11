import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs an exclusive cumulative sum (does not include the current element).

    Parameters:
        dim (int): The dimension along which to perform the exclusive cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # On the first run, capture the CUDA graph
            self.static_input = x.clone()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                exclusive_cumsum = torch.cat((torch.zeros_like(self.static_input.select(self.dim, 0).unsqueeze(self.dim)), self.static_input), dim=self.dim)[:-1]
                self.static_output = torch.cumsum(exclusive_cumsum, dim=self.dim)
        
        # For all runs, copy the new input data and replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        return self.static_output

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]