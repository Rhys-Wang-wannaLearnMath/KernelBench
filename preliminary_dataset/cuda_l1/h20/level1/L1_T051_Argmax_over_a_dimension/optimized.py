import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs Argmax over a specified dimension.
    
    Args:
        dim (int): The dimension to perform argmax over.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        
        # CUDA kernel for argmax along dimension 1
        self.cuda_kernel_code = '''
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <torch/extension.h>
        
        template <typename scalar_t>
        __global__ void argmax_dim1_kernel(
            const scalar_t* __restrict__ input,
            int64_t* __restrict__ output,
            const int batch_size,
            const int dim1,
            const int dim2) {
            
            // Each block processes one batch item and one column
            const int batch_idx = blockIdx.y;
            const int d2_idx = blockIdx.x;
            
            if (batch_idx >= batch_size || d2_idx >= dim2) return;
            
            // Thread ID within the block
            const int tid = threadIdx.x;
            
            // Calculate input base index for this batch and column
            const int base_idx = batch_idx * dim1 * dim2 + d2_idx;
            
            // Shared memory for reduction
            __shared__ scalar_t s_values[256];
            __shared__ int s_indices[256];
            
            // Each thread initializes with its value
            scalar_t thread_max_val = -INFINITY;
            int thread_max_idx = -1;
            
            // Each thread processes one or more elements
            for (int d1 = tid; d1 < dim1; d1 += blockDim.x) {
                const scalar_t val = input[base_idx + d1 * dim2];
                
                // Update max value and index if needed
                // Note: for equal values, keep the first occurrence (smaller index)
                if (val > thread_max_val || (val == thread_max_val && d1 < thread_max_idx) || thread_max_idx == -1) {
                    thread_max_val = val;
                    thread_max_idx = d1;
                }
            }
            
            // Store thread results in shared memory
            s_values[tid] = thread_max_val;
            s_indices[tid] = thread_max_idx;
            __syncthreads();
            
            // Parallel reduction in shared memory
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) {
                    scalar_t val1 = s_values[tid];
                    scalar_t val2 = s_values[tid + stride];
                    int idx1 = s_indices[tid];
                    int idx2 = s_indices[tid + stride];
                    
                    // Update if val2 is greater, or if equal and idx2 is smaller
                    if (val2 > val1 || (val2 == val1 && idx2 < idx1)) {
                        s_values[tid] = val2;
                        s_indices[tid] = idx2;
                    }
                }
                __syncthreads();
            }
            
            // Thread 0 writes the final result
            if (tid == 0) {
                output[batch_idx * dim2 + d2_idx] = s_indices[0];
            }
        }
        
        torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
            // Get input dimensions
            auto sizes = input.sizes();
            int ndim = sizes.size();
            
            // Validate dimension
            dim = dim < 0 ? dim + ndim : dim;
            TORCH_CHECK(dim >= 0 && dim < ndim, "Dimension out of range");
            
            // Currently only optimized for dim=1 with 3D tensors
            TORCH_CHECK(dim == 1 && ndim == 3, "This optimized kernel only supports dim=1 with 3D tensors");
            
            int batch_size = sizes[0];
            int dim1 = sizes[1];
            int dim2 = sizes[2];
            
            // Create output tensor with the dimension removed
            auto output = torch::empty({batch_size, dim2}, 
                                      torch::TensorOptions()
                                        .dtype(torch::kLong)
                                        .device(input.device()));
            
            // Launch kernel with optimized configuration
            const int threads_per_block = 256;
            dim3 grid(dim2, batch_size);
            
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_dim1_kernel", ([&] {
                argmax_dim1_kernel<scalar_t><<<grid, threads_per_block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<int64_t>(),
                    batch_size,
                    dim1,
                    dim2
                );
            }));
            
            return output;
        }
        '''
        
        # Compile the CUDA kernel if on GPU
        if torch.cuda.is_available():
            try:
                from torch.utils.cpp_extension import load_inline
                self.argmax_cuda = load_inline(
                    name="argmax_cuda",
                    cpp_sources="",
                    cuda_sources=self.cuda_kernel_code,
                    functions=["argmax_cuda"],
                    verbose=False
                )
            except Exception as e:
                print(f"Failed to compile CUDA kernel: {e}")
                self.argmax_cuda = None
        else:
            self.argmax_cuda = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies argmax over the specified dimension to the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with argmax applied, with the specified dimension removed.
        """
        # Use our custom CUDA kernel if available and input is on CUDA
        if self.argmax_cuda is not None and x.is_cuda and self.dim == 1 and x.dim() == 3:
            try:
                return self.argmax_cuda.argmax_cuda(x, self.dim)
            except Exception as e:
                # Fallback to PyTorch implementation if there's an error
                return torch.argmax(x, dim=self.dim)
        else:
            # Fall back to PyTorch implementation
            return torch.argmax(x, dim=self.dim)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]  # dim=1 as in the reference implementation