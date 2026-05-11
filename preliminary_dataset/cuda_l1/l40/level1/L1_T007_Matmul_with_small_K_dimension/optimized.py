import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation of matrix multiplication (C = A * B)
    with a small K dimension
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.output = None
        self.stream = None
        self.warmed_up = False
        self.device = None
        
        # Set optimal CUDA flags for performance
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.allow_tf32 = True
            self.device = torch.device('cuda')
            self.stream = torch.cuda.Stream()
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication.

        Args:
            A: Input tensor of shape (M, K).
            B: Input tensor of shape (K, N).

        Returns:
            Output tensor of shape (M, N).
        """
        # Ultra-fast path for the common case - already warmed up with tensors on GPU
        if self.warmed_up and A.is_cuda and B.is_cuda and self.output is not None:
            torch.matmul(A, B, out=self.output)
            return self.output
        
        # Ensure tensors are on GPU with non-blocking transfers
        if not A.is_cuda:
            if self.device is None:
                self.device = torch.device('cuda')
            A = A.to(self.device, non_blocking=True)
        elif self.device is None:
            self.device = A.device
            
        if not B.is_cuda:
            B = B.to(self.device, non_blocking=True)
        
        # Ensure tensors are contiguous for optimal memory access
        if not A.is_contiguous():
            A = A.contiguous()
        if not B.is_contiguous():
            B = B.contiguous()
        
        # Create or reuse output tensor
        M, K = A.shape
        K_b, N = B.shape
        
        if self.output is None or self.output.shape != (M, N) or self.output.device != A.device:
            self.output = torch.empty((M, N), dtype=A.dtype, device=A.device)
        
        # Create CUDA stream if not already created
        if self.stream is None:
            self.stream = torch.cuda.Stream()
        
        # Perform a warm-up run if not already done
        if not self.warmed_up:
            # Single efficient warm-up with a moderate-sized subset
            # This primes the GPU without excessive overhead
            torch.matmul(A[:256], B[:, :256], out=self.output[:256, :256])
            torch.cuda.synchronize()  # Synchronize only during warm-up
            self.warmed_up = True
        
        # Use PyTorch's built-in matmul with output tensor
        # No stream or with-context needed in the hot path for maximum performance
        torch.matmul(A, B, out=self.output)
        
        return self.output

M = 16384
N = 16384
K = 32

def get_inputs():
    A = torch.randn(M, K)
    B = torch.randn(K, N)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed