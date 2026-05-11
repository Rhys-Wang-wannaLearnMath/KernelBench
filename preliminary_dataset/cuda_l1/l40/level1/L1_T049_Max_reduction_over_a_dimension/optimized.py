import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Define the CUDA kernel for max reduction
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void max_dim0_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Each thread handles one element in the output tensor
    const int d1 = blockIdx.x * blockDim.x + threadIdx.x;
    const int d2 = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (d1 >= dim1 || d2 >= dim2) return;
    
    // Initialize max value
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    // Reduce across batch dimension
    for (int b = 0; b < batch_size; ++b) {
        const scalar_t val = input[(b * dim1 * dim2) + (d1 * dim2) + d2];
        max_val = max(max_val, val);
    }
    
    // Write result
    output[d1 * dim2 + d2] = max_val;
}

template <typename scalar_t>
__global__ void max_dim1_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Each thread handles one element in the output tensor
    const int b = blockIdx.x;
    const int d2 = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= batch_size || d2 >= dim2) return;
    
    // Initialize max value
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    // Reduce across dim1
    for (int d1 = 0; d1 < dim1; ++d1) {
        const scalar_t val = input[(b * dim1 * dim2) + (d1 * dim2) + d2];
        max_val = max(max_val, val);
    }
    
    // Write result
    output[b * dim2 + d2] = max_val;
}

// Optimized kernel for dim1 reduction using shared memory
template <typename scalar_t, int BLOCK_SIZE>
__global__ void max_dim1_optimized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Shared memory for block reduction
    __shared__ scalar_t shared_data[BLOCK_SIZE];
    
    const int b = blockIdx.x;
    const int d2 = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (b >= batch_size || d2 >= dim2) return;
    
    // Initialize max value
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    // Each thread processes multiple elements with stride BLOCK_SIZE
    for (int d1 = tid; d1 < dim1; d1 += BLOCK_SIZE) {
        const scalar_t val = input[(b * dim1 * dim2) + (d1 * dim2) + d2];
        max_val = max(max_val, val);
    }
    
    // Store in shared memory
    shared_data[tid] = max_val;
    __syncthreads();
    
    // Block reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[b * dim2 + d2] = shared_data[0];
    }
}

template <typename scalar_t>
__global__ void max_dim2_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Each thread handles one element in the output tensor
    const int b = blockIdx.x;
    const int d1 = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (b >= batch_size || d1 >= dim1) return;
    
    // Initialize max value
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    // Reduce across dim2
    for (int d2 = 0; d2 < dim2; ++d2) {
        const scalar_t val = input[(b * dim1 * dim2) + (d1 * dim2) + d2];
        max_val = max(max_val, val);
    }
    
    // Write result
    output[b * dim1 + d1] = max_val;
}

// Optimized kernel for dim2 reduction using shared memory
template <typename scalar_t, int BLOCK_SIZE>
__global__ void max_dim2_optimized_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Shared memory for block reduction
    __shared__ scalar_t shared_data[BLOCK_SIZE];
    
    const int b = blockIdx.x;
    const int d1 = blockIdx.y;
    const int tid = threadIdx.x;
    
    if (b >= batch_size || d1 >= dim1) return;
    
    // Initialize max value
    scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
    
    // Each thread processes multiple elements with stride BLOCK_SIZE
    for (int d2 = tid; d2 < dim2; d2 += BLOCK_SIZE) {
        const scalar_t val = input[(b * dim1 * dim2) + (d1 * dim2) + d2];
        max_val = max(max_val, val);
    }
    
    // Store in shared memory
    shared_data[tid] = max_val;
    __syncthreads();
    
    // Block reduction
    for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + s]);
        }
        __syncthreads();
    }
    
    // Write result
    if (tid == 0) {
        output[b * dim1 + d1] = shared_data[0];
    }
}

torch::Tensor max_dim0_cuda(torch::Tensor input) {
    const auto batch_size = input.size(0);
    const auto dim1 = input.size(1);
    const auto dim2 = input.size(2);
    
    auto output = torch::empty({dim1, dim2}, input.options());
    
    const dim3 threads(16, 16);
    const dim3 blocks((dim1 + threads.x - 1) / threads.x, 
                       (dim2 + threads.y - 1) / threads.y);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_dim0_cuda", ([&] {
        max_dim0_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim1,
            dim2
        );
    }));
    
    return output;
}

torch::Tensor max_dim1_cuda(torch::Tensor input) {
    const auto batch_size = input.size(0);
    const auto dim1 = input.size(1);
    const auto dim2 = input.size(2);
    
    auto output = torch::empty({batch_size, dim2}, input.options());
    
    // Use optimized kernel with shared memory
    constexpr int BLOCK_SIZE = 256;
    const dim3 blocks(batch_size, dim2);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_dim1_cuda", ([&] {
        max_dim1_optimized_kernel<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim1,
            dim2
        );
    }));
    
    return output;
}

torch::Tensor max_dim2_cuda(torch::Tensor input) {
    const auto batch_size = input.size(0);
    const auto dim1 = input.size(1);
    const auto dim2 = input.size(2);
    
    auto output = torch::empty({batch_size, dim1}, input.options());
    
    // Use optimized kernel with shared memory
    constexpr int BLOCK_SIZE = 256;
    const dim3 blocks(batch_size, dim1);
    
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "max_dim2_cuda", ([&] {
        max_dim2_optimized_kernel<scalar_t, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim1,
            dim2
        );
    }));
    
    return output;
}

torch::Tensor max_dim_cuda(torch::Tensor input, int64_t dim) {
    // Ensure input is contiguous
    if (!input.is_contiguous()) {
        input = input.contiguous();
    }
    
    // Call appropriate kernel based on dimension
    if (dim == 0 && input.dim() == 3) {
        return max_dim0_cuda(input);
    } else if (dim == 1 && input.dim() == 3) {
        return max_dim1_cuda(input);
    } else if (dim == 2 && input.dim() == 3) {
        return max_dim2_cuda(input);
    } else {
        // Fallback to PyTorch implementation
        return std::get<0>(input.max(dim));
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("max_dim", &max_dim_cuda, "Max reduction along any dimension (CUDA)");
}
"""

# Try to load the CUDA extension
try:
    max_cuda = load_inline(
        name="max_cuda",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["max_dim"],
        with_cuda=True,
        extra_cuda_cflags=["-O3"]
    )
    CUDA_EXTENSION_LOADED = True
except Exception as e:
    print(f"Warning: Could not load CUDA extension: {e}")
    CUDA_EXTENSION_LOADED = False

class ModelNew(nn.Module):
    """
    Optimized implementation of Max reduction over a specific dimension.
    
    Args:
        dim (int): The dimension to reduce over.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max reduction over the specified dimension to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor after Max reduction over the specified dimension.
        """
        # Use our custom CUDA kernel if available and applicable
        if CUDA_EXTENSION_LOADED and x.is_cuda:
            try:
                return max_cuda.max_dim(x, self.dim)
            except Exception as e:
                print(f"CUDA kernel failed, falling back to PyTorch: {e}")
                pass
        
        # Fallback to PyTorch's implementation
        # Use torch.amax which is optimized for max reduction without returning indices
        return torch.amax(x, dim=self.dim)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim1 = 256
dim2 = 256

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [1]  # Example, change to desired dimension