import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with a small K dimension
    """
    def __init__(self):
        super(Model, self).__init__()
        # Attributes for CUDA graph functionality
        self.graph = None
        self.static_A = None
        self.static_B = None
        self.static_C = None
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # On the first run, capture the CUDA graph
        if self.graph is None:
            # Create static placeholders for the graph's inputs and outputs.
            # Their memory addresses are fixed and are used by the graph.
            self.static_A = torch.empty_like(A)
            self.static_B = torch.empty_like(B)
            
            # Instantiate the graph
            self.graph = torch.cuda.CUDAGraph()

            # Enter graph capture context. All operations within this context
            # are recorded in the graph.
            with torch.cuda.graph(self.graph):
                # Define the graph's operations using the static placeholders
                self.static_C = torch.matmul(self.static_A, self.static_B)
        
        # Copy the current input data to the static placeholders
        self.static_A.copy_(A)
        self.static_B.copy_(B)
        
        # Replay the graph with the new input data
        self.graph.replay()
        
        # Return a clone of the static output tensor. Cloning is important to
        # avoid returning a reference to the graph's internal memory.
        return self.static_C.clone()

M = 16384
N = 16384
K = 32

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed