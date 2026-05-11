import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

class ModelNew(nn.Module):
    """
    An optimized model that performs a reverse cumulative sum operation along a specified dimension.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        self.cuda_module = None
        
        if torch.cuda.is_available():
            self._load_cuda_kernel()
    
    def _load_cuda_kernel(self):
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        
        template <typename scalar_t>
        __global__ void reverse_cumsum_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int seq_len) {
            
            // Each block processes one batch item
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;
            const int block_size = blockDim.x;
            
            if (batch_idx < batch_size) {
                // Get pointers to this batch's data
                const scalar_t* input_row = input + batch_idx * seq_len;
                scalar_t* output_row = output + batch_idx * seq_len;
                
                // Each thread processes multiple elements in a strided fashion
                // This ensures good memory coalescing
                for (int i = seq_len - 1 - tid; i >= 0; i -= block_size) {
                    scalar_t sum = 0;
                    
                    // Compute the reverse cumsum for this element
                    for (int j = i; j < seq_len; j++) {
                        sum += input_row[j];
                    }
                    
                    output_row[i] = sum;
                }
            }
        }
        
        // More efficient kernel for our specific sequence length
        template <typename scalar_t>
        __global__ void efficient_reverse_cumsum_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int seq_len) {
            
            // Each block processes one batch item
            const int batch_idx = blockIdx.x;
            const int tid = threadIdx.x;
            const int block_size = blockDim.x;
            
            if (batch_idx < batch_size) {
                // Get pointers to this batch's data
                const scalar_t* input_row = input + batch_idx * seq_len;
                scalar_t* output_row = output + batch_idx * seq_len;
                
                // Calculate the number of elements each thread will process
                const int elements_per_thread = (seq_len + block_size - 1) / block_size;
                const int start_idx = seq_len - 1 - tid * elements_per_thread;
                const int end_idx = max(-1, start_idx - elements_per_thread);
                
                // Process elements from end to beginning
                scalar_t running_sum = 0;
                for (int i = start_idx; i > end_idx; --i) {
                    if (i >= 0 && i < seq_len) {
                        running_sum += input_row[i];
                        output_row[i] = running_sum;
                    }
                }
            }
        }
        
        torch::Tensor reverse_cumsum_cuda_forward(
            torch::Tensor input,
            int dim) {
            
            // Only support dim=1 for now (as per the reference implementation)
            TORCH_CHECK(dim == 1, "Only dim=1 is currently supported");
            
            // Get input shape
            const auto batch_size = input.size(0);
            const auto seq_len = input.size(1);
            
            // Create output tensor
            auto output = torch::empty_like(input);
            
            // Choose kernel configuration based on sequence length
            const int threads_per_block = 128;
            const dim3 blocks(batch_size);
            const dim3 threads(threads_per_block);
            
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "reverse_cumsum_forward", ([&] {
                efficient_reverse_cumsum_kernel<scalar_t><<<blocks, threads>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    seq_len
                );
            }));
            
            return output;
        }
        
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &reverse_cumsum_cuda_forward, "Reverse CumSum forward (CUDA)");
        }
        """
        
        try:
            self.cuda_module = load_inline(
                name="reverse_cumsum_cuda",
                cpp_sources="",
                cuda_sources=cuda_source,
                functions=["forward"],
                verbose=False,
                with_cuda=True,
                extra_cuda_cflags=["-O3"]
            )
        except Exception as e:
            print(f"Failed to compile CUDA extension: {e}")
            self.cuda_module = None

    def forward(self, x):
        # Use custom CUDA kernel if available and input is on GPU
        if self.cuda_module is not None and x.is_cuda and self.dim == 1:
            return self.cuda_module.forward(x.contiguous(), self.dim)
        else:
            # Fall back to PyTorch implementation
            return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]