import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a matrix multiplication (C = A * B) where A and B are lower triangular matrices.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.cuda_kernel = None
        try:
            self.cuda_kernel = self._load_cuda_kernel()
        except Exception as e:
            print(f"Warning: CUDA kernel compilation failed: {e}. Falling back to PyTorch implementation.")
    
    def _load_cuda_kernel(self):
        cuda_code = '''
        extern "C" __global__ void triangular_matmul(
            const float* __restrict__ A,
            const float* __restrict__ B,
            float* __restrict__ C,
            const int N)
        {
            // Block size for shared memory
            const int TILE_SIZE = 32;
            
            // Shared memory for tiles of A and B with padding to avoid bank conflicts
            __shared__ float As[TILE_SIZE][TILE_SIZE+1];
            __shared__ float Bs[TILE_SIZE][TILE_SIZE+1];
            
            // Global row and column indices
            int row = blockIdx.y * TILE_SIZE + threadIdx.y;
            int col = blockIdx.x * TILE_SIZE + threadIdx.x;
            
            // Only compute elements in the lower triangular part
            if (row >= col && row < N && col < N) {
                // Register cache for accumulated sum
                float sum = 0.0f;
                
                // Calculate the range of k values we need to consider
                // For triangular matrices, we only need k from col to row
                int k_global_start = col;
                int k_global_end = min(row + 1, N);
                
                // Loop over tiles
                for (int t = k_global_start / TILE_SIZE; t <= (k_global_end - 1) / TILE_SIZE; ++t) {
                    int k_base = t * TILE_SIZE;
                    
                    // Initialize shared memory
                    As[threadIdx.y][threadIdx.x] = 0.0f;
                    Bs[threadIdx.y][threadIdx.x] = 0.0f;
                    __syncthreads();
                    
                    // Load A[row, k] where k <= row (since A is lower triangular)
                    int k = k_base + threadIdx.x;
                    if (k < N && row >= k && k >= k_global_start && k < k_global_end) {
                        As[threadIdx.y][threadIdx.x] = A[row * N + k];
                    }
                    
                    // Load B[k, col] where k >= col (since B is lower triangular)
                    k = k_base + threadIdx.y;
                    if (k < N && k >= col && k < k_global_end) {
                        Bs[threadIdx.y][threadIdx.x] = B[k * N + col];
                    }
                    __syncthreads();
                    
                    // Compute partial dot product using the tiles
                    // Only consider k values from max(col, k_base) to min(row, k_base + TILE_SIZE - 1)
                    int k_start = max(k_global_start - k_base, 0);
                    int k_end = min(k_global_end - k_base, TILE_SIZE);
                    
                    // Aggressive loop unrolling for better instruction-level parallelism
                    #pragma unroll 16
                    for (int k = k_start; k < k_end; ++k) {
                        sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
                    }
                    __syncthreads();
                }
                
                // Write result to global memory
                C[row * N + col] = sum;
            }
        }
        '''
        
        from torch.utils.cpp_extension import load_inline
        return load_inline(
            name='triangular_matmul_kernel',
            cpp_sources='',
            cuda_sources=cuda_code,
            functions=['triangular_matmul'],
            with_cuda=True,
            verbose=False
        )
    
    def _optimized_pytorch_impl(self, A, B):
        """
        Fallback implementation using PyTorch operations.
        """
        N = A.shape[0]
        device = A.device
        dtype = A.dtype
        
        # Pre-allocate result matrix
        C = torch.zeros((N, N), dtype=dtype, device=device)
        
        # Set block size based on matrix size
        block_size = 1024
        
        # Process in column-major order for better memory access patterns
        for j in range(0, N, block_size):
            j_end = min(j + block_size, N)
            
            # Process only the lower triangular blocks
            for i in range(j, N, block_size):
                i_end = min(i + block_size, N)
                
                # For this output block C[i:i_end, j:j_end], we need to compute
                # sum over k of A[i:i_end, k] * B[k, j:j_end]
                # But we only need k from j to i_end due to triangular structure
                
                k_start = j
                k_end = i_end
                
                if k_start < k_end:
                    # Extract the relevant portions of A and B for this computation
                    A_slice = A[i:i_end, k_start:k_end]
                    B_slice = B[k_start:k_end, j:j_end]
                    
                    # Perform the matrix multiplication for this block
                    C[i:i_end, j:j_end] = torch.matmul(A_slice, B_slice)
        
        return C
    
    def forward(self, A, B):
        """
        Performs optimized matrix multiplication of lower triangular matrices A and B.

        Args:
            A (torch.Tensor): Lower triangular matrix of shape (N, N).
            B (torch.Tensor): Lower triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The result of matrix multiplication C of shape (N, N).
        """
        N = A.shape[0]
        
        # For small matrices or if CUDA kernel failed to load, use PyTorch implementation
        if N <= 128 or self.cuda_kernel is None or not A.is_cuda:
            if N <= 128:
                return torch.tril(torch.matmul(A, B))
            else:
                return self._optimized_pytorch_impl(A, B)
        
        # Use our custom CUDA kernel for large matrices
        device = A.device
        dtype = A.dtype
        
        # Create output tensor
        C = torch.zeros((N, N), dtype=dtype, device=device)
        
        # Set grid and block dimensions
        threads_per_block = 32
        blocks_per_grid_x = (N + threads_per_block - 1) // threads_per_block
        blocks_per_grid_y = (N + threads_per_block - 1) // threads_per_block
        
        # Launch the kernel
        self.cuda_kernel.triangular_matmul(
            grid=(blocks_per_grid_x, blocks_per_grid_y),
            block=(threads_per_block, threads_per_block),
            args=[A.contiguous().data_ptr(), B.contiguous().data_ptr(), C.data_ptr(), N]
        )
        
        return C

M = 4096

def get_inputs():
    A = torch.randn(M, M)
    B = torch.randn(M, M)
    A = torch.tril(A)
    B = torch.tril(B)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed