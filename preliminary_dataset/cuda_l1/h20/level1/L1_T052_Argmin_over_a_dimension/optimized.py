import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

# Define CUDA kernel code
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void argmin_kernel(
    const scalar_t* __restrict__ input,
    int64_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Calculate global indices
    const int batch_idx = blockIdx.y;
    const int dim2_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Check if this thread should process data
    if (batch_idx < batch_size && dim2_idx < dim2) {
        // Calculate base index for this thread's data
        const int base_idx = batch_idx * dim1 * dim2 + dim2_idx;
        
        // Initialize with first element
        scalar_t min_val = input[base_idx];
        int min_idx = 0;
        
        // Process elements in groups of 8 for better instruction-level parallelism
        int i = 1;
        
        // Main loop with aggressive unrolling
        for (; i + 7 < dim1; i += 8) {
            // Prefetch next batch of values (compiler hint)
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                __builtin_prefetch(&input[base_idx + (i + j + 8) * dim2], 0, 0);
            }
            
            // Load 8 values at once
            const scalar_t val1 = input[base_idx + i * dim2];
            const scalar_t val2 = input[base_idx + (i+1) * dim2];
            const scalar_t val3 = input[base_idx + (i+2) * dim2];
            const scalar_t val4 = input[base_idx + (i+3) * dim2];
            const scalar_t val5 = input[base_idx + (i+4) * dim2];
            const scalar_t val6 = input[base_idx + (i+5) * dim2];
            const scalar_t val7 = input[base_idx + (i+6) * dim2];
            const scalar_t val8 = input[base_idx + (i+7) * dim2];
            
            // Streamlined comparison approach with minimal branching
            if (val1 < min_val) { min_val = val1; min_idx = i; }
            if (val2 < min_val) { min_val = val2; min_idx = i+1; }
            if (val3 < min_val) { min_val = val3; min_idx = i+2; }
            if (val4 < min_val) { min_val = val4; min_idx = i+3; }
            if (val5 < min_val) { min_val = val5; min_idx = i+4; }
            if (val6 < min_val) { min_val = val6; min_idx = i+5; }
            if (val7 < min_val) { min_val = val7; min_idx = i+6; }
            if (val8 < min_val) { min_val = val8; min_idx = i+7; }
        }
        
        // Handle remaining elements
        for (; i < dim1; ++i) {
            const scalar_t val = input[base_idx + i * dim2];
            if (val < min_val) {
                min_val = val;
                min_idx = i;
            }
        }
        
        // Write result to output
        output[batch_idx * dim2 + dim2_idx] = min_idx;
    }
}

// Alternative kernel with different thread block configuration
template <typename scalar_t>
__global__ void argmin_kernel_alt(
    const scalar_t* __restrict__ input,
    int64_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Calculate global indices - different block organization
    // Each warp handles 32 consecutive elements in dim2
    const int batch_idx = blockIdx.z;
    const int warp_group = blockIdx.y * blockDim.y + threadIdx.y;
    const int lane_idx = threadIdx.x;
    const int dim2_idx = blockIdx.x * 32 + lane_idx;
    
    // Check if this thread should process data
    if (batch_idx < batch_size && warp_group == 0 && dim2_idx < dim2) {
        // Calculate base index for this thread's data
        const int base_idx = batch_idx * dim1 * dim2 + dim2_idx;
        
        // Initialize with first element
        scalar_t min_val = input[base_idx];
        int min_idx = 0;
        
        // Process elements with a different unrolling strategy
        // This can help with memory access patterns on some GPU architectures
        int i = 1;
        for (; i + 3 < dim1; i += 4) {
            const scalar_t val1 = input[base_idx + i * dim2];
            const scalar_t val2 = input[base_idx + (i+1) * dim2];
            const scalar_t val3 = input[base_idx + (i+2) * dim2];
            const scalar_t val4 = input[base_idx + (i+3) * dim2];
            
            if (val1 < min_val) { min_val = val1; min_idx = i; }
            if (val2 < min_val) { min_val = val2; min_idx = i+1; }
            if (val3 < min_val) { min_val = val3; min_idx = i+2; }
            if (val4 < min_val) { min_val = val4; min_idx = i+3; }
        }
        
        // Handle remaining elements
        for (; i < dim1; ++i) {
            const scalar_t val = input[base_idx + i * dim2];
            if (val < min_val) {
                min_val = val;
                min_idx = i;
            }
        }
        
        // Write result to output
        output[batch_idx * dim2 + dim2_idx] = min_idx;
    }
}

torch::Tensor argmin_cuda(torch::Tensor input, int dim) {
    // Check that we're reducing along dimension 1
    TORCH_CHECK(dim == 1, "Custom CUDA kernel only supports reduction along dimension 1");
    
    // Get tensor dimensions
    const auto batch_size = input.size(0);
    const auto dim1 = input.size(1);
    const auto dim2 = input.size(2);
    
    // Create output tensor
    auto output = torch::empty({batch_size, dim2}, 
                              torch::TensorOptions()
                                  .dtype(torch::kLong)
                                  .device(input.device()));
    
    // Calculate grid and block dimensions for main kernel
    const int threads_per_block = 256;
    const dim3 blocks(
        (dim2 + threads_per_block - 1) / threads_per_block,
        batch_size
    );
    const dim3 threads(threads_per_block);
    
    // Calculate grid and block dimensions for alternative kernel
    const dim3 alt_blocks(
        (dim2 + 31) / 32,
        1,
        batch_size
    );
    const dim3 alt_threads(32, 1);
    
    // Choose kernel based on dimensions
    // For our specific dimensions (batch_size=16, dim1=256, dim2=256),
    // the main kernel should be more efficient
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "argmin_cuda", ([&] {
        if (dim2 <= 512) {
            argmin_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<int64_t>(),
                batch_size,
                dim1,
                dim2
            );
        } else {
            argmin_kernel_alt<scalar_t><<<alt_blocks, alt_threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<int64_t>(),
                batch_size,
                dim1,
                dim2
            );
        }
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("argmin", &argmin_cuda, "Argmin operation along dimension 1 (CUDA)");
}
"""

# Compile the CUDA extension
try:
    argmin_cuda = load_inline(
        name="argmin_cuda_ext",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["argmin"],
        with_cuda=True,
        extra_cuda_cflags=["-O3", "--use_fast_math"]
    )
except Exception as e:
    # Fallback if compilation fails
    argmin_cuda = None
    print(f"Failed to compile CUDA extension: {e}")

class ModelNew(nn.Module):
    """
    Optimized implementation of argmin along a specified dimension using CUDA.
    
    Args:
        dim (int): Dimension along which to find the minimum value.
    """
    def __init__(self, dim: int):
        super(ModelNew, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Finds the index of the minimum value along the specified dimension.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Tensor containing the indices of the minimum values along the specified dimension.
        """
        # Use PyTorch's built-in argmin if:
        # 1. Our CUDA extension failed to compile
        # 2. Input is not on CUDA
        # 3. Dimension is not 1
        # 4. Input doesn't have exactly 3 dimensions
        if (argmin_cuda is None or not x.is_cuda or self.dim != 1 or x.dim() != 3):
            return torch.argmin(x, dim=self.dim)
        
        # Use our custom CUDA kernel
        try:
            # Move tensor to contiguous memory layout if it's not already
            if not x.is_contiguous():
                x = x.contiguous()
                
            return argmin_cuda.argmin(x, self.dim)
        except Exception as e:
            # Fallback to PyTorch implementation if our kernel fails
            print(f"Custom kernel failed, falling back to PyTorch: {e}")
            return torch.argmin(x, dim=self.dim)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim1 = 256
dim2 = 256
dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [dim]