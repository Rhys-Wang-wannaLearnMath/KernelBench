import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(Model, self).__init__()
        self.dim = dim
        
        # Attributes for CUDA graph optimization
        self.graph = None
        self.static_input = None
        self.static_output = None
        # A non-default stream is required for graph capture
        self.stream = torch.cuda.Stream()

    def forward(self, x):
        # On the first forward pass, capture the model's operations in a CUDA graph.
        if self.graph is None:
            # Create a placeholder for the input tensor with the same properties as the real input.
            self.static_input = torch.empty_like(x)
            
            # Instantiate the CUDA graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Begin graph capture on the non-default stream.
            with torch.cuda.graph(self.graph, stream=self.stream):
                # Define the operations to be captured using the static placeholder.
                # The output of this sequence of operations becomes the graph's static output.
                self.static_output = torch.cumsum(self.static_input.flip(self.dim), dim=self.dim).flip(self.dim)
        
        # On every forward pass (including the first), copy the current input data
        # into the static input placeholder.
        self.static_input.copy_(x)
        
        # Replay the captured graph to execute the operations.
        self.graph.replay()
        
        # Return the static output tensor, which now contains the result of the graph execution.
        return self.static_output

batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]