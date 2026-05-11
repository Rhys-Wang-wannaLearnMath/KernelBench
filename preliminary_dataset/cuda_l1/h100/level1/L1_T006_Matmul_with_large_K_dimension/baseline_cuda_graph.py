import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a large K dimension
    """
    def __init__(self):
        super(Model, self).__init__()
        self.graph = None
        self.static_A = None
        self.static_B = None
        self.static_C = None
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication of A and B.

        Args:
            A: Input tensor of shape (M, K)
            B: Input tensor of shape (K, N)

        Returns:
            Output tensor of shape (M, N)
        """
        # On the first forward pass, capture the CUDA graph
        if self.graph is None:
            # Create static tensors to hold the input shapes and types
            self.static_A = torch.empty_like(A)
            self.static_B = torch.empty_like(B)
            
            # Instantiate a new CUDA graph
            g = torch.cuda.CUDAGraph()
            
            # Begin capturing operations into the graph
            with torch.cuda.graph(g):
                self.static_C = torch.matmul(self.static_A, self.static_B)
            
            # Save the captured graph for future replays
            self.graph = g

        # Copy the current input data into the static tensors
        self.static_A.copy_(A)
        self.static_B.copy_(B)
        
        # Replay the captured graph with the new input data
        self.graph.replay()
        
        # Return the result tensor from the graph's static output
        return self.static_C

M = 256
N = 256
K = 131072

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed