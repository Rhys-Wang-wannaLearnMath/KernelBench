import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B)
    with optimized memory access patterns
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (K, M).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Ensure tensors are contiguous for optimal memory access
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        
        # Use the mathematical identity (A.T @ B) = (B.T @ A).T
        # This avoids the explicit transpose operation and has better memory access patterns
        # Use torch.mm for direct matrix multiplication (more efficient than matmul for 2D tensors)
        return torch.mm(B.T, A).T

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
M = 1024
K = 4096
N = 2048

def get_inputs():
    A = torch.randn(K, M)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed