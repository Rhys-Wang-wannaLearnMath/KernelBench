import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    """
    def __init__(self):
        super(Model, self).__init__()
        # Dictionary to hold cudnn backend flags.
        # Example: self.cudnn_flags = {'benchmark': True, 'deterministic': False}
        self.cudnn_flags = {}
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Apply cudnn backend flags if they are set
        if self.cudnn_flags:
            with torch.backends.cudnn.flags(**self.cudnn_flags):
                return torch.matmul(A, B.T)
        else:
            return torch.matmul(A, B.T)

M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(N, K)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed