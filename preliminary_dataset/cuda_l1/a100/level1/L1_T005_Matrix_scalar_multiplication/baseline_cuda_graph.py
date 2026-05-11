import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix-scalar multiplication (C = A * s)
    """
    def __init__(self):
        super(Model, self).__init__()
        # Placeholders for the CUDA graph and static tensors
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, A: torch.Tensor, s: float) -> torch.Tensor:
        """
        Performs matrix-scalar multiplication.

        Args:
            A: Input matrix of shape (M, N)
            s: Scalar value

        Returns:
            C: Resulting matrix of shape (M, N)
        """
        # On the first run, the graph is not yet captured.
        if self.graph is None:
            # Create a static copy of the input tensor. CUDA graphs require
            # static memory addresses for inputs and outputs.
            self.static_input = A.clone()
            
            # Instantiate and capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # All operations within this block are recorded in the graph.
                # The output tensor is created and becomes part of the graph.
                self.static_output = self.static_input * s
        
        # For every run (including the first), copy the current input data
        # to the static input tensor.
        self.static_input.copy_(A)
        
        # Replay the captured graph. This executes the recorded CUDA kernels
        # with minimal CPU overhead.
        self.graph.replay()
        
        # Return the static output tensor which now holds the result.
        return self.static_output

M = 16384
N = 4096

def get_inputs():
    A = torch.randn(M, N)
    s = 3.14
    return [A, s]

def get_init_inputs():
    return []  # No special initialization inputs needed