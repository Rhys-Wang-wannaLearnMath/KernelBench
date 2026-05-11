import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

class ModelNew(nn.Module):
    """
    A model that performs a reverse cumulative sum operation along a specified dimension.
    This implementation uses an optimized custom CUDA kernel for improved performance.

    Parameters:
        dim (int): The dimension along which to perform the reverse cumulative sum.
    """

    def __init__(self, dim):
        super(ModelNew, self).__init__()
        self.dim = dim
        
        # CUDA kernel for reverse cumsum
        cuda_source = """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>

        #define WARP_SIZE 32
        #define FULL_MASK 0xffffffff

        template <typename scalar_t>
        __global__ void reverse_cumsum_kernel(
            const scalar_t* __restrict__ input,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int seq_len) {
            
            // Each block handles one batch element
            const int batch_idx = blockIdx.x;
            if (batch_idx >= batch_size) return;
            
            // Get pointers to the current batch element
            const scalar_t* batch_input = input + batch_idx * seq_len;
            scalar_t* batch_output = output + batch_idx * seq_len;
            
            // Thread index within the block
            const int tid = threadIdx.x;
            const int block_size = blockDim.x;
            
            // Compute reverse cumulative sum directly
            // Start from the end of the sequence (right-to-left)
            scalar_t running_sum = 0;
            
            // Each thread processes elements in a strided fashion
            for (int i = seq_len - 1 - tid; i >= 0; i -= block_size) {
                running_sum += batch_input[i];
                batch_output[i] = running_sum;
            }
        }

        torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim) {
            // Ensure input is contiguous
            input = input.contiguous();
            
            // Get dimensions
            const int batch_size = input.size(0);
            const int seq_len = input.size(1);
            
            // Create output tensor
            auto output = torch::empty_like(input);
            
            // Ensure dim is valid
            if (dim != 1) {
                throw std::runtime_error("Only dim=1 is supported in this kernel");
            }
            
            // Calculate thread block size
            const int threads_per_block = 256;
            
            // Launch kernel
            AT_DISPATCH_FLOATING_TYPES(input.type(), "reverse_cumsum_cuda", ([&] {
                reverse_cumsum_kernel<scalar_t><<<batch_size, threads_per_block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    seq_len
                );
            }));
            
            return output;
        }
        """

        cpp_source = """
        #include <torch/extension.h>

        torch::Tensor reverse_cumsum_cuda(torch::Tensor input, int dim);

        torch::Tensor reverse_cumsum(torch::Tensor input, int dim) {
            if (input.device().is_cuda()) {
                return reverse_cumsum_cuda(input, dim);
            } else {
                // Fall back to CPU implementation
                return torch::cumsum(input.flip(dim), dim).flip(dim);
            }
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("reverse_cumsum", &reverse_cumsum, "Reverse cumsum operation");
        }
        """
        
        # Compile the CUDA extension
        self.cuda_module = None
        try:
            # Use a unique name to avoid conflicts
            extension_name = f"reverse_cumsum_cuda_{os.getpid()}"
            self.cuda_module = load_inline(
                name=extension_name,
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                functions=["reverse_cumsum"],
                verbose=False
            )
        except Exception as e:
            print(f"Failed to compile CUDA extension: {e}, falling back to PyTorch implementation")

    def forward(self, x):
        if self.cuda_module is not None and x.is_cuda and self.dim == 1:
            try:
                return self.cuda_module.reverse_cumsum(x, self.dim)
            except Exception as e:
                print(f"CUDA kernel failed: {e}, falling back to PyTorch implementation")
                return torch.cumsum(x.flip(self.dim), dim=self.dim).flip(self.dim)
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