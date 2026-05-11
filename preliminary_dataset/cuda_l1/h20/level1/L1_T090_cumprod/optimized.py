import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Define the CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA kernel for cumulative product along dimension 1
template <typename scalar_t>
__global__ void cumprod_dim1_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim_size) {
    
    // Each thread handles one row in the batch
    const int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Starting index for this batch element
        const int start_idx = batch_idx * dim_size;
        
        // Copy first element as is
        output[start_idx] = input[start_idx];
        
        // Compute cumulative product for the rest of the elements
        for (int i = 1; i < dim_size; ++i) {
            output[start_idx + i] = output[start_idx + i - 1] * input[start_idx + i];
        }
    }
}

// CUDA kernel for cumulative product along dimension 1 with shared memory
template <typename scalar_t>
__global__ void cumprod_dim1_shared_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim_size) {
    
    extern __shared__ char shared_memory[];
    scalar_t* shared = reinterpret_cast<scalar_t*>(shared_memory);
    
    const int batch_idx = blockIdx.x;
    const int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Starting index for this batch element
        const int start_idx = batch_idx * dim_size;
        
        // Load data into shared memory
        for (int i = tid; i < dim_size; i += blockDim.x) {
            shared[i] = input[start_idx + i];
        }
        __syncthreads();
        
        // Compute cumulative product in shared memory
        for (int stride = 1; stride < dim_size; stride *= 2) {
            __syncthreads();
            for (int i = tid; i < dim_size; i += blockDim.x) {
                if (i >= stride) {
                    shared[i] = shared[i] * shared[i - stride];
                }
            }
        }
        __syncthreads();
        
        // Write results back to global memory
        for (int i = tid; i < dim_size; i += blockDim.x) {
            output[start_idx + i] = shared[i];
        }
    }
}

// Launch the appropriate kernel based on the input size
torch::Tensor cumprod_cuda(torch::Tensor input, int64_t dim) {
    auto output = torch::empty_like(input);
    
    // Get tensor dimensions
    auto sizes = input.sizes();
    int64_t batch_size = sizes[0];
    int64_t dim_size = sizes[1];
    
    // Choose kernel based on dimension size
    if (dim == 1) {
        const int threads_per_block = 256;
        const int blocks = (batch_size + threads_per_block - 1) / threads_per_block;
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "cumprod_cuda", ([&] {
            if (dim_size <= 1024) {
                // For smaller dimensions, use the simpler kernel
                cumprod_dim1_kernel<scalar_t><<<blocks, threads_per_block>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    dim_size
                );
            } else {
                // For larger dimensions, use shared memory kernel
                const int threads = 256;
                const int shared_mem_size = dim_size * sizeof(scalar_t);
                cumprod_dim1_shared_kernel<scalar_t><<<batch_size, threads, shared_mem_size>>>(
                    input.data_ptr<scalar_t>(),
                    output.data_ptr<scalar_t>(),
                    batch_size,
                    dim_size
                );
            }
        }));
    } else {
        // For other dimensions, fall back to PyTorch's implementation
        output = torch::cumprod(input, dim);
    }
    
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor cumprod_cuda(torch::Tensor input, int64_t dim);

torch::Tensor cumprod(torch::Tensor input, int64_t dim) {
    if (input.device().is_cuda()) {
        return cumprod_cuda(input, dim);
    } else {
        return torch::cumprod(input, dim);
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cumprod", &cumprod, "Optimized cumulative product");
}
"""

# Compile the CUDA extension
try:
    cumprod_cuda = load_inline(
        name="cumprod_cuda",
        cpp_sources=[cpp_source],
        cuda_sources=[cuda_source],
        functions=["cumprod"],
        verbose=True,
        with_cuda=True,
        build_directory=os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions")
    )
except Exception as e:
    print(f"Failed to compile CUDA extension: {e}")
    # Fallback to PyTorch's implementation
    cumprod_cuda = None

class ModelNew(nn.Module):
    """
    A model that performs a cumulative product operation along a specified dimension
    with an optimized CUDA implementation.

    Parameters:
        dim (int): The dimension along which to perform the cumulative product operation.
    """

    def __init__(self, dim):
        """
        Initialize the CumulativeProductModel.

        Args:
            dim (int): The dimension along which to perform the cumulative product.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.output = None
        self.use_custom_kernel = cumprod_cuda is not None

    def forward(self, x):
        """
        Forward pass, computing the cumulative product along the specified dimension
        using an optimized CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, *input_shape).

        Returns:
            torch.Tensor: Tensor of the same shape as `x` after applying cumulative product along `dim`.
        """
        if self.output is None:
            self.output = torch.empty_like(x)
        
        if self.use_custom_kernel and x.is_cuda:
            # Use our optimized CUDA kernel
            return cumprod_cuda.cumprod(x, self.dim)
        else:
            # Fall back to PyTorch's implementation with pre-allocated output
            return torch.cumprod(x, dim=self.dim, out=self.output)

# Define input dimensions and parameters
batch_size = 128
input_shape = (4000,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [dim]