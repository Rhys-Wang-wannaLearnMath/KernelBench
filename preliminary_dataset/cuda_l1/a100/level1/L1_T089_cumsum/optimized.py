import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    """
    A simple model that performs a cumulative sum (prefix sum) operation along a specified dimension.
    Optimized implementation using custom CUDA kernels.

    Parameters:
        dim (int): The dimension along which to perform the scan operation.
    """

    def __init__(self, dim):
        """
        Initialize the Scan model.

        Args:
            dim (int): The dimension along which to perform the cumulative sum.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self._output_buffer = None
        self._initialized = False
        self._cuda_module = None
        
        # Initialize CUDA module if CUDA is available
        if torch.cuda.is_available():
            self._initialize_cuda_module()

    def _initialize_cuda_module(self):
        """Initialize the CUDA module with our custom kernel."""
        try:
            from torch.utils.cpp_extension import load_inline
            
            # CUDA kernel for cumulative sum
            cuda_source = """
            #include <torch/extension.h>
            #include <cuda.h>
            #include <cuda_runtime.h>
            
            template <typename scalar_t>
            __global__ void cumsum_kernel(
                const scalar_t* __restrict__ input,
                scalar_t* __restrict__ output,
                const int batch_size,
                const int seq_len) {
                
                // Get batch index
                const int batch_idx = blockIdx.x;
                
                // Return if out of bounds
                if (batch_idx >= batch_size) return;
                
                // Get pointers to current batch
                const scalar_t* batch_input = input + batch_idx * seq_len;
                scalar_t* batch_output = output + batch_idx * seq_len;
                
                // Use shared memory for efficient access
                extern __shared__ char shared_mem[];
                scalar_t* temp = reinterpret_cast<scalar_t*>(shared_mem);
                
                // Each thread loads one or more elements into shared memory
                for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
                    temp[i] = batch_input[i];
                }
                __syncthreads();
                
                // Perform parallel scan in shared memory (Blelloch scan algorithm)
                // Up-sweep phase (reduce)
                int stride = 1;
                while (stride < seq_len) {
                    int index = (threadIdx.x + 1) * stride * 2 - 1;
                    if (index < seq_len && (index + stride) < seq_len) {
                        temp[index + stride] += temp[index];
                    }
                    stride *= 2;
                    __syncthreads();
                }
                
                // Down-sweep phase (distribute)
                if (threadIdx.x == 0) {
                    temp[seq_len - 1] = batch_input[seq_len - 1]; // Restore the last element
                }
                __syncthreads();
                
                stride = seq_len / 2;
                while (stride > 0) {
                    int index = (threadIdx.x + 1) * stride * 2 - 1;
                    if (index + stride < seq_len) {
                        scalar_t t = temp[index];
                        temp[index] = (index == 0) ? batch_input[0] : temp[index - stride];
                        temp[index + stride] += t;
                    }
                    stride /= 2;
                    __syncthreads();
                }
                
                // Write results back to global memory
                for (int i = threadIdx.x; i < seq_len; i += blockDim.x) {
                    batch_output[i] = temp[i];
                }
            }
            
            // Optimized kernel for large sequences that don't fit in shared memory
            template <typename scalar_t>
            __global__ void cumsum_large_kernel(
                const scalar_t* __restrict__ input,
                scalar_t* __restrict__ output,
                const int batch_size,
                const int seq_len) {
                
                // Get batch index
                const int batch_idx = blockIdx.x;
                
                // Return if out of bounds
                if (batch_idx >= batch_size) return;
                
                // Get pointers to current batch
                const scalar_t* batch_input = input + batch_idx * seq_len;
                scalar_t* batch_output = output + batch_idx * seq_len;
                
                // First element is copied as is
                if (threadIdx.x == 0) {
                    batch_output[0] = batch_input[0];
                }
                
                // Each thread computes a partial sum for its segment
                const int segment_size = (seq_len + blockDim.x - 1) / blockDim.x;
                const int start_idx = threadIdx.x * segment_size;
                const int end_idx = min(start_idx + segment_size, seq_len);
                
                if (start_idx < seq_len) {
                    // First element of segment gets its value from the previous segment's last element
                    scalar_t sum = batch_input[start_idx];
                    batch_output[start_idx] = sum;
                    
                    // Compute partial sums within this segment
                    for (int i = start_idx + 1; i < end_idx; i++) {
                        sum += batch_input[i];
                        batch_output[i] = sum;
                    }
                }
                __syncthreads();
                
                // Now we need to add the prefix sums from previous segments
                for (int stride = 1; stride < blockDim.x; stride *= 2) {
                    int idx = threadIdx.x;
                    int src_idx = idx - stride;
                    
                    if (idx >= stride && start_idx < seq_len) {
                        int src_end_idx = min((src_idx + 1) * segment_size - 1, seq_len - 1);
                        scalar_t prefix_sum = batch_output[src_end_idx];
                        
                        for (int i = start_idx; i < end_idx; i++) {
                            batch_output[i] += prefix_sum;
                        }
                    }
                    __syncthreads();
                }
            }
            
            torch::Tensor cumsum_cuda(torch::Tensor input, int dim) {
                // Only support dim=1 for now
                TORCH_CHECK(dim == 1, "Only dim=1 is supported for now");
                TORCH_CHECK(input.dim() == 2, "Input must be 2D tensor");
                
                const auto batch_size = input.size(0);
                const auto seq_len = input.size(1);
                
                auto output = torch::empty_like(input);
                
                // Configure kernel parameters
                const int threads = 256;
                const int blocks = batch_size;
                const int shared_mem_size = seq_len * sizeof(float);
                
                // Choose the appropriate kernel based on sequence length and available shared memory
                AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumsum_cuda", ([&] {
                    // Check if we can fit the sequence in shared memory
                    if (shared_mem_size <= 48 * 1024) { // 48KB is typical max shared memory per block
                        cumsum_kernel<scalar_t><<<blocks, threads, shared_mem_size>>>(
                            input.data_ptr<scalar_t>(),
                            output.data_ptr<scalar_t>(),
                            batch_size,
                            seq_len
                        );
                    } else {
                        // Use the large sequence kernel that doesn't rely on shared memory as much
                        cumsum_large_kernel<scalar_t><<<blocks, threads>>>(
                            input.data_ptr<scalar_t>(),
                            output.data_ptr<scalar_t>(),
                            batch_size,
                            seq_len
                        );
                    }
                }));
                
                return output;
            }
            
            PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                m.def("cumsum", &cumsum_cuda, "Cumulative sum CUDA implementation");
            }
            """
            
            self._cuda_module = load_inline(
                name="cumsum_cuda",
                cpp_sources="",
                cuda_sources=cuda_source,
                functions=["cumsum"],
                verbose=False
            )
            self._initialized = True
        except Exception as e:
            # Fallback if compilation fails
            self._initialized = False
            print(f"Warning: Failed to compile CUDA extension: {e}")

    def forward(self, x):
        """
        Forward pass for the Scan model, computing the cumulative sum along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape), where `*input_shape` 
                              can vary depending on the use case.

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative sum along `dim`.
        """
        # Fast path for the common case (dim=1, 2D tensor)
        if (self._initialized and x.is_cuda and x.dim() == 2 and 
            self.dim == 1 and x.is_contiguous()):
            try:
                return self._cuda_module.cumsum(x, self.dim)
            except Exception:
                # Fall back to PyTorch implementation if CUDA kernel fails
                pass
        
        # Fallback to PyTorch's implementation with buffer reuse
        if x.is_cuda:
            if self._output_buffer is None or self._output_buffer.shape != x.shape or self._output_buffer.device != x.device:
                self._output_buffer = torch.empty_like(x)
            
            # Use PyTorch's cumsum with preallocated output buffer
            return torch.cumsum(x, dim=self.dim, out=self._output_buffer)
        else:
            # For CPU tensors, just use PyTorch's implementation
            return torch.cumsum(x, dim=self.dim)


# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)  # Example shape (arbitrary)
dim = 1

def get_inputs():
    """
    Generates random inputs for testing the Scan model.

    Returns:
        list: A list containing a single randomly generated tensor with shape 
              (batch_size, *input_shape).
    """
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    """
    Returns the initialization parameters for the Scan model.

    Returns:
        list: A list containing the `dim` parameter for model initialization.
    """
    return [dim]