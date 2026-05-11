import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Define CUDA kernel for optimized diagonal matrix multiplication
cuda_source = """
extern "C" __global__ void diagonal_matmul_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int N,
    const int M) {
    
    // Calculate global thread indices
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if thread is within bounds
    if (row < N && col < M) {
        // Get the diagonal element for this row
        const float a_val = A[row];
        
        // Calculate output index
        const int idx = row * M + col;
        
        // Perform the multiplication
        C[idx] = a_val * B[idx];
    }
}
"""

# Try to compile the CUDA kernel
try:
    diagonal_matmul_module = load_inline(
        name="diagonal_matmul_cuda",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["diagonal_matmul_kernel"],
        with_cuda=True,
        extra_cuda_cflags=["-O3"],
        verbose=False
    )
    CUDA_KERNEL_AVAILABLE = True
except Exception:
    CUDA_KERNEL_AVAILABLE = False

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
        # Get dimensions
        N, M = B.shape
        
        # Check if we can use the CUDA kernel
        if (CUDA_KERNEL_AVAILABLE and A.is_cuda and B.is_cuda and 
            A.is_contiguous() and B.is_contiguous() and 
            A.dtype == torch.float32 and B.dtype == torch.float32):
            
            # Allocate output tensor
            C = torch.empty_like(B)
            
            # Configure kernel launch parameters
            threads_per_block = 32
            blocks_x = (M + threads_per_block - 1) // threads_per_block
            blocks_y = (N + threads_per_block - 1) // threads_per_block
            
            # Launch the kernel
            diagonal_matmul_module.diagonal_matmul_kernel(
                grid=(blocks_x, blocks_y),
                block=(threads_per_block, threads_per_block),
                args=[A.data_ptr(), B.data_ptr(), C.data_ptr(), N, M]
            )
            
            return C
        else:
            # Fallback to PyTorch implementation
            return B * A.unsqueeze(1)

M = 4096
N = 4096

def get_inputs():
    A = torch.randn(N)
    B = torch.randn(N, M)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed