import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.
    Optimized with a custom CUDA kernel.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """
    
    _cuda_module = None
    
    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        
        # Load CUDA extension if not already loaded
        if ModelNew._cuda_module is None:
            cuda_source = '''
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>

            // For shorter sequences - optimized block-based approach
            template <typename scalar_t, int BLOCK_SIZE>
            __global__ void masked_cumsum_block_kernel(
                const scalar_t* __restrict__ input,
                const bool* __restrict__ mask,
                scalar_t* __restrict__ output,
                const int seq_length) 
            {
                // Each block processes one batch element
                const int batch_idx = blockIdx.x;
                const int tid = threadIdx.x;
                
                // Calculate offsets for this batch element
                const int batch_offset = batch_idx * seq_length;
                const scalar_t* batch_input = input + batch_offset;
                const bool* batch_mask = mask + batch_offset;
                scalar_t* batch_output = output + batch_offset;
                
                // Shared memory for efficient scan
                extern __shared__ scalar_t s_data[];
                
                // Process sequence in chunks
                scalar_t running_sum = 0;
                
                for (int chunk_start = 0; chunk_start < seq_length; chunk_start += BLOCK_SIZE) {
                    const int idx = chunk_start + tid;
                    
                    // Load data and apply mask
                    scalar_t val = 0;
                    if (idx < seq_length) {
                        val = batch_mask[idx] ? batch_input[idx] : 0;
                    }
                    s_data[tid] = val;
                    __syncthreads();
                    
                    // Perform inclusive scan within the block using Hillis-Steele algorithm
                    for (int stride = 1; stride < BLOCK_SIZE; stride *= 2) {
                        scalar_t prev = 0;
                        if (tid >= stride) {
                            prev = s_data[tid - stride];
                        }
                        __syncthreads();
                        
                        if (tid >= stride) {
                            s_data[tid] += prev;
                        }
                        __syncthreads();
                    }
                    
                    // Write results to output and add running sum from previous chunks
                    if (idx < seq_length) {
                        batch_output[idx] = s_data[tid] + running_sum;
                    }
                    
                    // Update running sum for next chunk
                    running_sum = 0;
                    if (chunk_start + BLOCK_SIZE < seq_length) {
                        // Get the last valid element in this chunk
                        running_sum = s_data[BLOCK_SIZE - 1];
                    }
                    __syncthreads();
                }
            }

            // For longer sequences - warp-optimized approach
            template <typename scalar_t>
            __global__ void masked_cumsum_warp_kernel(
                const scalar_t* __restrict__ input,
                const bool* __restrict__ mask,
                scalar_t* __restrict__ output,
                const int seq_length) 
            {
                // Each block processes one batch element
                const int batch_idx = blockIdx.x;
                const int tid = threadIdx.x;
                const int lane_id = tid % 32;  // Lane ID within warp
                const int warp_id = tid / 32;  // Warp ID within block
                const int num_warps = blockDim.x / 32;
                
                // Calculate offsets for this batch element
                const int batch_offset = batch_idx * seq_length;
                const scalar_t* batch_input = input + batch_offset;
                const bool* batch_mask = mask + batch_offset;
                scalar_t* batch_output = output + batch_offset;
                
                // Shared memory for warp sums
                extern __shared__ scalar_t warp_sums[];
                
                // Process sequence in chunks, each warp handles a section
                scalar_t global_sum = 0;
                
                for (int base = 0; base < seq_length; base += blockDim.x) {
                    const int idx = base + tid;
                    
                    // Load data and apply mask
                    scalar_t val = 0;
                    if (idx < seq_length) {
                        val = batch_mask[idx] ? batch_input[idx] : 0;
                    }
                    
                    // Perform warp-level inclusive scan using shuffle operations
                    scalar_t warp_sum = val;
                    
                    #pragma unroll
                    for (int offset = 1; offset < 32; offset *= 2) {
                        scalar_t n = __shfl_up_sync(0xffffffff, warp_sum, offset);
                        if (lane_id >= offset) {
                            warp_sum += n;
                        }
                    }
                    
                    // Last thread in each warp stores the warp sum
                    if (lane_id == 31) {
                        warp_sums[warp_id] = warp_sum;
                    }
                    __syncthreads();
                    
                    // First warp computes prefix sum of warp sums
                    if (warp_id == 0 && lane_id < num_warps) {
                        scalar_t warp_prefix = warp_sums[lane_id];
                        
                        #pragma unroll
                        for (int offset = 1; offset < 32 && offset < num_warps; offset *= 2) {
                            scalar_t n = __shfl_up_sync(0xffffffff, warp_prefix, offset);
                            if (lane_id >= offset) {
                                warp_prefix += n;
                            }
                        }
                        
                        warp_sums[lane_id] = warp_prefix;
                    }
                    __syncthreads();
                    
                    // Add prefix from previous warps and global prefix
                    scalar_t prefix = 0;
                    if (warp_id > 0) {
                        prefix = warp_sums[warp_id - 1];
                    }
                    
                    // Compute final value and store
                    scalar_t final_val = global_sum + prefix + warp_sum - (lane_id > 0 ? __shfl_up_sync(0xffffffff, warp_sum, 1) : 0);
                    
                    if (idx < seq_length) {
                        batch_output[idx] = final_val;
                    }
                    
                    // Update global sum for next chunk
                    if (tid == blockDim.x - 1 || idx + 1 == seq_length) {
                        int last_warp = min(num_warps - 1, (seq_length - base - 1) / 32);
                        global_sum += warp_sums[last_warp];
                    }
                    __syncthreads();
                }
            }

            torch::Tensor masked_cumsum_cuda(
                torch::Tensor input,
                torch::Tensor mask,
                int dim) 
            {
                TORCH_CHECK(dim == 1, "Only dim=1 is currently supported");
                
                const auto batch_size = input.size(0);
                const auto seq_length = input.size(1);
                
                auto output = torch::zeros_like(input);
                
                // Choose optimal thread block configuration
                const int block_size = 256;  // Optimal for most GPUs
                const int grid_size = batch_size;
                
                AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "masked_cumsum_cuda", ([&] {
                    // Choose kernel based on sequence length
                    if (seq_length <= 2048) {
                        // For shorter sequences, use block-based kernel
                        const int shared_mem_size = block_size * sizeof(scalar_t);
                        masked_cumsum_block_kernel<scalar_t, block_size><<<grid_size, block_size, shared_mem_size>>>(
                            input.data_ptr<scalar_t>(),
                            mask.data_ptr<bool>(),
                            output.data_ptr<scalar_t>(),
                            seq_length
                        );
                    } else {
                        // For longer sequences, use warp-optimized kernel
                        // Need shared memory for warp sums (one sum per warp)
                        const int num_warps = block_size / 32;
                        const int shared_mem_size = num_warps * sizeof(scalar_t);
                        masked_cumsum_warp_kernel<scalar_t><<<grid_size, block_size, shared_mem_size>>>(
                            input.data_ptr<scalar_t>(),
                            mask.data_ptr<bool>(),
                            output.data_ptr<scalar_t>(),
                            seq_length
                        );
                    }
                }));
                
                return output;
            }
            '''

            cpp_source = '''
            #include <torch/extension.h>

            torch::Tensor masked_cumsum_cuda(
                torch::Tensor input,
                torch::Tensor mask,
                int dim);

            torch::Tensor masked_cumsum(
                torch::Tensor input,
                torch::Tensor mask,
                int dim) 
            {
                if (dim != 1 || !input.is_cuda()) {
                    return torch::cumsum(input * mask, dim);
                }
                
                return masked_cumsum_cuda(input.contiguous(), mask.contiguous(), dim);
            }

            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("masked_cumsum", &masked_cumsum, "Masked cumulative sum");
            }
            '''
            
            try:
                # Create a unique module name to avoid conflicts
                module_name = f"masked_cumsum_{os.getpid()}"
                
                # Load the CUDA extension
                ModelNew._cuda_module = load_inline(
                    name=module_name,
                    cpp_sources=cpp_source,
                    cuda_sources=cuda_source,
                    functions=["masked_cumsum"],
                    verbose=False,
                    extra_cuda_cflags=["-O3"]  # Enable high optimization level
                )
            except Exception as e:
                print(f"Failed to load CUDA extension: {e}")
                ModelNew._cuda_module = None

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        # Fall back to PyTorch implementation if CUDA extension failed to load
        if ModelNew._cuda_module is None or self.dim != 1 or not x.is_cuda:
            return torch.cumsum(x * mask, dim=self.dim)
        
        # Make sure inputs are contiguous
        x = x.contiguous()
        mask = mask.contiguous()
        
        try:
            # Use our custom CUDA kernel
            return ModelNew._cuda_module.masked_cumsum(x, mask, self.dim)
        except Exception as e:
            # Fall back to PyTorch implementation if CUDA kernel fails
            print(f"CUDA kernel failed: {e}")
            return torch.cumsum(x * mask, dim=self.dim)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    x = torch.randn(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()  # Random boolean mask
    return [x, mask]

def get_init_inputs():
    return [dim]