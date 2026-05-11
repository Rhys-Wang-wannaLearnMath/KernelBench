import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
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
        if self.graph is None:
            # On the first call, capture the CUDA graph
            self.graph = torch.cuda.CUDAGraph()
            self.static_A = torch.zeros_like(A)
            self.static_B = torch.zeros_like(B)
            
            with torch.cuda.graph(self.graph):
                self.static_C = torch.matmul(self.static_A.T, self.static_B)
        
        # Copy the current input data to the static tensors
        self.static_A.copy_(A)
        self.static_B.copy_(B)
        
        # Replay the graph with the new data
        self.graph.replay()
        
        return self.static_C

M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(K, M)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed