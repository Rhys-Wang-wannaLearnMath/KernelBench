import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    """
    A model that performs a masked cumulative sum, only summing elements that satisfy a condition.
    Optimized with a custom CUDA kernel using an efficient parallel scan algorithm.

    Parameters:
        dim (int): The dimension along which to perform the masked cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.kernel = None
        
        if torch.cuda.is_available():
            self._init_cuda_kernel()
    
    def _init_cuda_kernel(self):
        cuda_source = '''
        #include <cuda_runtime.h>
        
        #define WARP_SIZE 32
        #define ELEMENTS_PER_THREAD 8
        
        extern "C" __global__ void masked_cumsum_kernel(
            const float* __restrict__ input,
            const bool* __restrict__ mask,
            float* __restrict__ output,
            const int seq_len)
        {
            // Each block processes one row (batch item)
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;
            const int num_threads = blockDim.x;
            
            // Calculate starting index for this batch
            const int batch_offset = batch_idx * seq_len;
            
            // Shared memory for scan operation
            extern __shared__ float s_data[];
            
            // Phase 1: Each thread processes multiple elements sequentially
            float running_sum = 0.0f;
            
            // Calculate the range of elements this thread will process
            const int elements_per_thread = (seq_len + num_threads - 1) / num_threads;
            const int start_idx = tid * elements_per_thread;
            const int end_idx = min(start_idx + elements_per_thread, seq_len);
            
            // Process assigned elements sequentially
            for (int i = start_idx; i < end_idx; i++) {
                const int global_idx = batch_offset + i;
                const float val = mask[global_idx] ? input[global_idx] : 0.0f;
                running_sum += val;
                output[global_idx] = running_sum;
            }
            
            // Store the final sum for this thread in shared memory
            s_data[tid] = running_sum;
            __syncthreads();
            
            // Phase 2: Parallel prefix sum on the thread sums
            // This is an exclusive scan to compute the offset for each thread's section
            for (int stride = 1; stride < num_threads; stride *= 2) {
                float val = 0.0f;
                if (tid >= stride) {
                    val = s_data[tid - stride];
                }
                __syncthreads();
                
                if (tid >= stride) {
                    s_data[tid] += val;
                }
                __syncthreads();
            }
            
            // Phase 3: Update each thread's section with the offset
            if (tid > 0 && start_idx < seq_len) {
                const float offset = s_data[tid - 1];
                for (int i = start_idx; i < end_idx; i++) {
                    output[batch_offset + i] += offset;
                }
            }
        }
        
        extern "C" __global__ void optimized_masked_cumsum_kernel(
            const float* __restrict__ input,
            const bool* __restrict__ mask,
            float* __restrict__ output,
            const int seq_len)
        {
            // Each block processes one row (batch item)
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;
            const int lane_id = tid % WARP_SIZE;
            const int warp_id = tid / WARP_SIZE;
            const int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
            
            // Calculate starting index for this batch
            const int batch_offset = batch_idx * seq_len;
            
            // Shared memory for warp sums
            extern __shared__ float s_warp_sums[];
            
            // Each thread processes multiple elements
            const int elements_per_thread = ELEMENTS_PER_THREAD;
            float thread_sum = 0.0f;
            
            // Process elements in chunks
            for (int base = 0; base < seq_len; base += blockDim.x * elements_per_thread) {
                // Each thread processes multiple elements sequentially
                for (int i = 0; i < elements_per_thread; i++) {
                    const int idx = base + tid + i * blockDim.x;
                    if (idx < seq_len) {
                        const int global_idx = batch_offset + idx;
                        const float val = mask[global_idx] ? input[global_idx] : 0.0f;
                        thread_sum += val;
                        output[global_idx] = thread_sum;
                    }
                }
                
                // Compute prefix sum within each warp
                float warp_sum = thread_sum;
                for (int offset = 1; offset < WARP_SIZE; offset *= 2) {
                    float val = __shfl_up_sync(0xffffffff, warp_sum, offset);
                    if (lane_id >= offset) {
                        warp_sum += val;
                    }
                }
                
                // Last thread in each warp writes the warp's sum to shared memory
                if (lane_id == WARP_SIZE - 1) {
                    s_warp_sums[warp_id] = warp_sum;
                }
                __syncthreads();
                
                // First warp computes prefix sum of warp sums
                if (warp_id == 0 && lane_id < num_warps) {
                    float warp_sum_val = s_warp_sums[lane_id];
                    for (int offset = 1; offset < num_warps; offset *= 2) {
                        float val = __shfl_up_sync(0xffffffff, warp_sum_val, offset);
                        if (lane_id >= offset) {
                            warp_sum_val += val;
                        }
                    }
                    s_warp_sums[lane_id] = warp_sum_val;
                }
                __syncthreads();
                
                // Add the prefix sum to all threads except the first warp
                if (warp_id > 0) {
                    thread_sum += s_warp_sums[warp_id - 1];
                    
                    // Update the output values for this thread
                    for (int i = 0; i < elements_per_thread; i++) {
                        const int idx = base + tid + i * blockDim.x;
                        if (idx < seq_len) {
                            output[batch_offset + idx] += s_warp_sums[warp_id - 1];
                        }
                    }
                }
                __syncthreads();
            }
        }
        '''
        
        try:
            self.kernel = load_inline(
                name="masked_cumsum_cuda",
                cpp_sources="",
                cuda_sources=cuda_source,
                functions=["masked_cumsum_kernel", "optimized_masked_cumsum_kernel"],
                with_cuda=True,
                verbose=False
            )
        except Exception as e:
            print(f"Failed to compile CUDA kernel: {e}")
            self.kernel = None

    def forward(self, x, mask):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).
            mask (torch.Tensor): Boolean mask of the same shape as x.

        Returns:
            torch.Tensor: Cumulative sum of elements where mask is True.
        """
        # Check if we can use our optimized implementation
        if (self.kernel is not None and x.is_cuda and mask.is_cuda and 
            self.dim == 1 and x.dim() == 2 and x.dtype == torch.float32):
            
            batch_size, seq_len = x.shape
            output = torch.empty_like(x)
            
            try:
                # Choose the appropriate kernel based on sequence length
                if seq_len <= 2048:
                    # For smaller sequences, use the simpler kernel
                    block_size = min(256, seq_len)
                    shared_mem_size = block_size * 4  # 4 bytes per float
                    
                    self.kernel.masked_cumsum_kernel(
                        grid=(batch_size, 1, 1),
                        block=(block_size, 1, 1),
                        args=[x.contiguous().data_ptr(), 
                              mask.contiguous().data_ptr(),
                              output.data_ptr(),
                              seq_len],
                        shared=shared_mem_size
                    )
                else:
                    # For larger sequences, use the optimized kernel
                    block_size = 256
                    num_warps = (block_size + 31) // 32
                    shared_mem_size = num_warps * 4  # 4 bytes per float
                    
                    self.kernel.optimized_masked_cumsum_kernel(
                        grid=(batch_size, 1, 1),
                        block=(block_size, 1, 1),
                        args=[x.contiguous().data_ptr(), 
                              mask.contiguous().data_ptr(),
                              output.data_ptr(),
                              seq_len],
                        shared=shared_mem_size
                    )
                
                return output
            except Exception as e:
                # Fall back to PyTorch implementation if kernel execution fails
                print(f"CUDA kernel execution failed: {e}")
                return torch.cumsum(x * mask, dim=self.dim)
        else:
            # Fall back to PyTorch implementation for other cases
            return torch.cumsum(x * mask, dim=self.dim)


# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    x = torch.randn(batch_size, *input_shape)
    mask = torch.randint(0, 2, x.shape).bool()  # Random boolean mask
    return [x, mask]

def get_init_inputs():
    return [dim]