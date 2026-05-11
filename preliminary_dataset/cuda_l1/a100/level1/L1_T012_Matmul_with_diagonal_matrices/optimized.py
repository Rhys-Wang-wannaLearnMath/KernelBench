import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication of a diagonal matrix with another matrix.
    C = diag(A) * B
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, A, B):
        """
        Performs the matrix multiplication.

        Args:
            A (torch.Tensor): A 1D tensor representing the diagonal of the diagonal matrix. Shape: (N,).
            B (torch.Tensor): A 2D tensor representing the second matrix. Shape: (N, M).

        Returns:
            torch.Tensor: The result of the matrix multiplication. Shape: (N, M).
        """
        # Mathematically equivalent to torch.diag(A) @ B but much more efficient
        # This avoids creating the full diagonal matrix
        # A.unsqueeze(1) converts A from shape (N,) to (N,1) for broadcasting
        return B * A.unsqueeze(1)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
M = 4096
N = 4096

def get_inputs():
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed