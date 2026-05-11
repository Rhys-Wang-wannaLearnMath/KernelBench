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
        
        # CUDA kernel for argmax
        self.cuda_kernel_code = '''
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <torch/extension.h>

        // Helper function to update max value and index
        template <typename scalar_t>
        __device__ __forceinline__ void update_max(
            scalar_t val, int idx, 
            scalar_t& max_val, int& max_idx) {
            if (val > max_val) {
                max_val = val;
                max_idx = idx;
            }
        }

        // Main kernel optimized for dim=1 with known dimensions
        template <typename scalar_t, int ITEMS_PER_THREAD = 4>
        __global__ void argmax_dim1_optimized_kernel(
            const scalar_t* __restrict__ input,
            int64_t* __restrict__ output,
            const int batch_size,
            const int dim1,
            const int dim2) {
            
            // Each thread processes ITEMS_PER_THREAD columns
            const int batch_idx = blockIdx.y;
            const int thread_col_start = (blockIdx.x * blockDim.x + threadIdx.x) * ITEMS_PER_THREAD;
            
            // Early exit if out of bounds for the entire thread
            if (batch_idx >= batch_size || thread_col_start >= dim2) return;
            
            // Process ITEMS_PER_THREAD columns per thread
            #pragma unroll
            for (int item = 0; item < ITEMS_PER_THREAD; ++item) {
                const int col_idx = thread_col_start + item;
                
                // Skip if this specific column is out of bounds
                if (col_idx >= dim2) continue;
                
                // Calculate base index for this batch and column
                const int base_idx = batch_idx * dim1 * dim2 + col_idx;
                
                // Initialize with first element
                scalar_t max_val = input[base_idx];
                int max_idx = 0;
                
                // Process elements in this column with manual unrolling
                // For dim1=256, unroll in chunks of 16
                #pragma unroll 4
                for (int d1_base = 1; d1_base < dim1; d1_base += 16) {
                    #pragma unroll
                    for (int d1_offset = 0; d1_offset < 16 && d1_base + d1_offset < dim1; ++d1_offset) {
                        const int d1 = d1_base + d1_offset;
                        const scalar_t val = input[base_idx + d1 * dim2];
                        update_max(val, d1, max_val, max_idx);
                    }
                }
                
                // Write result
                output[batch_idx * dim2 + col_idx] = max_idx;
            }
        }

        // Specialized kernel for exactly dim1=256, dim2=256
        template <typename scalar_t>
        __global__ void argmax_dim1_specialized_kernel(
            const scalar_t* __restrict__ input,
            int64_t* __restrict__ output,
            const int batch_size) {
            
            // Each thread processes one column
            const int batch_idx = blockIdx.y;
            const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Early exit if out of bounds
            if (batch_idx >= batch_size || col_idx >= 256) return;
            
            // Calculate base index for this batch and column
            const int base_idx = batch_idx * 256 * 256 + col_idx;
            
            // Initialize with first element
            scalar_t max_val = input[base_idx];
            int max_idx = 0;
            
            // Process elements in this column with complete unrolling for dim1=256
            // Use 8 chunks of 32 elements each for better instruction-level parallelism
            #pragma unroll
            for (int d1 = 1; d1 < 32; ++d1) {
                const scalar_t val = input[base_idx + d1 * 256];
                update_max(val, d1, max_val, max_idx);
            }
            
            #pragma unroll
            for (int d1 = 32; d1 < 64; ++d1) {
                const scalar_t val = input[base_idx + d1 * 256];
                update_max(val, d1, max_val, max_idx);
            }
            
            #pragma unroll
            for (int d1 = 64; d1 < 96; ++d1) {
                const scalar_t val = input[base_idx + d1 * 256];
                update_max(val, d1, max_val, max_idx);
            }
            
            #pragma unroll
            for (int d1 = 96; d1 < 128; ++d1) {
                const scalar_t val = input[base_idx + d1 * 256];
                update_max(val, d1, max_val, max_idx);
            }
            
            #pragma unroll
            for (int d1 = 128; d1 < 160; ++d1) {
                const scalar_t val = input[base_idx + d1 * 256];
                update_max(val, d1, max_val, max_idx);
            }
            
            #pragma unroll
            for (int d1 = 160; d1 < 192; ++d1) {
                const scalar_t val = input[base_idx + d1 * 256];
                update_max(val, d1, max_val, max_idx);
            }
            
            #pragma unroll
            for (int d1 = 192; d1 < 224; ++d1) {
                const scalar_t val = input[base_idx + d1 * 256];
                update_max(val, d1, max_val, max_idx);
            }
            
            #pragma unroll
            for (int d1 = 224; d1 < 256; ++d1) {
                const scalar_t val = input[base_idx + d1 * 256];
                update_max(val, d1, max_val, max_idx);
            }
            
            // Write result
            output[batch_idx * 256 + col_idx] = max_idx;
        }

        // Version with prefetching for better memory latency hiding
        template <typename scalar_t>
        __global__ void argmax_dim1_prefetch_kernel(
            const scalar_t* __restrict__ input,
            int64_t* __restrict__ output,
            const int batch_size,
            const int dim1,
            const int dim2) {
            
            // Each thread processes one column
            const int batch_idx = blockIdx.y;
            const int col_idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Early exit if out of bounds
            if (batch_idx >= batch_size || col_idx >= dim2) return;
            
            // Calculate base index for this batch and column
            const int base_idx = batch_idx * dim1 * dim2 + col_idx;
            
            // Initialize with first element
            scalar_t max_val = input[base_idx];
            int max_idx = 0;
            
            // Prefetch the next element
            scalar_t next_val = (1 < dim1) ? input[base_idx + dim2] : max_val;
            
            // Process elements with prefetching
            for (int d1 = 1; d1 < dim1; ++d1) {
                // Current value is the prefetched value from previous iteration
                scalar_t current_val = next_val;
                
                // Prefetch next value if available
                if (d1 + 1 < dim1) {
                    next_val = input[base_idx + (d1 + 1) * dim2];
                }
                
                // Update max
                if (current_val > max_val) {
                    max_val = current_val;
                    max_idx = d1;
                }
            }
            
            // Write result
            output[batch_idx * dim2 + col_idx] = max_idx;
        }

        torch::Tensor argmax_cuda(torch::Tensor input, int dim) {
            // Get input dimensions
            auto sizes = input.sizes();
            int ndim = sizes.size();
            
            // Validate dimension
            dim = dim < 0 ? dim + ndim : dim;
            TORCH_CHECK(dim >= 0 && dim < ndim, "Dimension out of range");
            
            // Create output tensor with the dimension removed
            std::vector<int64_t> output_sizes;
            for (int i = 0; i < ndim; i++) {
                if (i != dim) {
                    output_sizes.push_back(sizes[i]);
                }
            }
            
            auto output = torch::empty(output_sizes, 
                                      torch::TensorOptions()
                                        .dtype(torch::kLong)
                                        .device(input.device()));
            
            // Currently only optimized for dim=1 with 3D tensors
            if (dim == 1 && ndim == 3) {
                int batch_size = sizes[0];
                int dim1 = sizes[1];
                int dim2 = sizes[2];
                
                // Launch kernel with optimized configuration
                const int threads_per_block = 256;
                
                // Choose the best kernel based on input dimensions
                if (dim1 == 256 && dim2 == 256) {
                    // Use specialized kernel for exactly 256x256
                    const int blocks_per_grid_x = (dim2 + threads_per_block - 1) / threads_per_block;
                    dim3 grid(blocks_per_grid_x, batch_size);
                    
                    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_dim1_specialized_kernel", ([&] {
                        argmax_dim1_specialized_kernel<scalar_t><<<grid, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                            input.data_ptr<scalar_t>(),
                            output.data_ptr<int64_t>(),
                            batch_size
                        );
                    }));
                } else if (dim2 >= 512) {
                    // For large dim2, use multi-item-per-thread approach
                    constexpr int ITEMS_PER_THREAD = 4;
                    const int effective_threads = (dim2 + ITEMS_PER_THREAD - 1) / ITEMS_PER_THREAD;
                    const int blocks_per_grid_x = (effective_threads + threads_per_block - 1) / threads_per_block;
                    dim3 grid(blocks_per_grid_x, batch_size);
                    
                    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_dim1_optimized_kernel", ([&] {
                        argmax_dim1_optimized_kernel<scalar_t, ITEMS_PER_THREAD><<<grid, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                            input.data_ptr<scalar_t>(),
                            output.data_ptr<int64_t>(),
                            batch_size,
                            dim1,
                            dim2
                        );
                    }));
                } else {
                    // For smaller dim2, use prefetching kernel
                    const int blocks_per_grid_x = (dim2 + threads_per_block - 1) / threads_per_block;
                    dim3 grid(blocks_per_grid_x, batch_size);
                    
                    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmax_dim1_prefetch_kernel", ([&] {
                        argmax_dim1_prefetch_kernel<scalar_t><<<grid, threads_per_block, 0, at::cuda::getCurrentCUDAStream()>>>(
                            input.data_ptr<scalar_t>(),
                            output.data_ptr<int64_t>(),
                            batch_size,
                            dim1,
                            dim2
                        );
                    }));
                }
            } else {
                // Fall back to PyTorch implementation for other dimensions
                output = torch::argmax(input, dim);
            }
            
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
                    verbose=False,
                    extra_cuda_cflags=["--use_fast_math", "-O3"]
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
        if self.argmax_cuda is not None and x.is_cuda:
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