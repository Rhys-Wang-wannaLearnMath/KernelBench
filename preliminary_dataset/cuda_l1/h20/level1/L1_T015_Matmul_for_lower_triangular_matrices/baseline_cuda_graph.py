import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices. 
    """
    def __init__(self):
        super(Model, self).__init__()
        self.graph = None
        self.static_A = None
        self.static_B = None
        self.static_C = None
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        if self.graph is None:
            # On the first run, we create static tensors and capture the graph.
            # These static tensors will have fixed memory addresses.
            self.static_A = A.clone()
            self.static_B = B.clone()
            
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_C = torch.tril(torch.matmul(self.static_A, self.static_B))

        # For every run (including the first), copy the new input data into our static tensors.
        self.static_A.copy_(A)
        self.static_B.copy_(B)
        
        # Replay the captured graph.
        self.graph.replay()
        
        return self.static_C

M = 4096

def get_inputs():
    A = torch.randn(M, M)
    B = torch.randn(M, M)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed