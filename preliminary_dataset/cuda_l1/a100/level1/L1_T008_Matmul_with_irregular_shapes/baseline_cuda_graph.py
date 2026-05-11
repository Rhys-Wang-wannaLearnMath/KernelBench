import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with irregular shapes
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
            A: Input tensor with shape (M, K).
            B: Input tensor with shape (K, N).

        Returns:
            C: Output tensor with shape (M, N).
        """
        if self.graph is None:
            # First run: create static placeholders and record the graph.
            # Create placeholders with the same shape/dtype/device as the inputs.
            self.static_A = torch.empty_like(A)
            self.static_B = torch.empty_like(B)
            
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # The graph's operations are defined using the static placeholders.
                self.static_C = torch.matmul(self.static_A, self.static_B)

        # For every run, copy the current input data into the static placeholders.
        self.static_A.copy_(A)
        self.static_B.copy_(B)
        
        # Replay the graph. This executes the matmul with the updated data
        # and stores the result in self.static_C.
        self.graph.replay()
        
        return self.static_C

M = 8205
K = 2949
N = 5921

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed