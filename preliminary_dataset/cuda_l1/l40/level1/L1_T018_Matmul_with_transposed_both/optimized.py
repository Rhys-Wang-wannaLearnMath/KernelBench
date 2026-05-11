import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    with optimized implementation
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (N, K).

        Returns:
            Output tensor of shape (M, N).
        """
        # Ensure tensors are contiguous for optimal performance
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        
        # Use the mathematical identity (A.T @ B.T) = (B @ A).T
        # This avoids creating explicit transposed copies
        result = torch.matmul(B, A).transpose(0, 1)
        
        return result

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(K, M)
    B = torch.randn(N, K)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed