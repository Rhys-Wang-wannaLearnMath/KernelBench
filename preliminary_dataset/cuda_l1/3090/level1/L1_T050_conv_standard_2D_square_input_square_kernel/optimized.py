import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void product_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Use 2D thread block structure for better occupancy
    // Each warp (32 threads) processes one column
    const int warp_size = 32;
    const int batch_idx = blockIdx.z;
    const int col_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane_id = threadIdx.x;
    
    if (batch_idx < batch_size && col_idx < dim2) {
        // Phase 1: Fast zero detection
        bool has_zero = false;
        
        // Each thread checks a portion of the column with stride warp_size
        for (int i = lane_id; i < dim1 && !has_zero; i += warp_size) {
            scalar_t val = input[batch_idx * dim1 * dim2 + i * dim2 + col_idx];
            has_zero = (val == 0.0f);
        }
        
        // Use warp vote to determine if any thread found a zero
        unsigned int mask = __activemask();
        bool warp_has_zero = __any_sync(mask, has_zero);
        
        if (warp_has_zero) {
            // Early termination - result is zero
            if (lane_id == 0) {
                output[batch_idx * dim2 + col_idx] = 0.0f;
            }
            return;
        }
        
        // Phase 2: Efficient product computation with multiple accumulators
        scalar_t acc1 = 1.0f;
        scalar_t acc2 = 1.0f;
        scalar_t acc3 = 1.0f;
        scalar_t acc4 = 1.0f;
        
        // Process elements in chunks of 4 with stride warp_size
        int i = lane_id;
        const int stride = warp_size * 4;
        
        // Main loop - process 4 elements at a time with stride warp_size
        for (; i + 3*warp_size < dim1; i += stride) {
            acc1 *= input[batch_idx * dim1 * dim2 + i * dim2 + col_idx];
            acc2 *= input[batch_idx * dim1 * dim2 + (i + warp_size) * dim2 + col_idx];
            acc3 *= input[batch_idx * dim1 * dim2 + (i + 2*warp_size) * dim2 + col_idx];
            acc4 *= input[batch_idx * dim1 * dim2 + (i + 3*warp_size) * dim2 + col_idx];
        }
        
        // Handle remaining elements
        for (; i < dim1; i += warp_size) {
            acc1 *= input[batch_idx * dim1 * dim2 + i * dim2 + col_idx];
        }
        
        // Combine accumulators
        scalar_t thread_product = acc1 * acc2 * acc3 * acc4;
        
        // Warp-level reduction using shuffle operations
        for (int offset = warp_size/2; offset > 0; offset /= 2) {
            thread_product *= __shfl_down_sync(mask, thread_product, offset);
        }
        
        // Write final result
        if (lane_id == 0) {
            output[batch_idx * dim2 + col_idx] = thread_product;
        }
    }
}

torch::Tensor product_reduction_cuda(torch::Tensor input, int dim) {
    // Get tensor dimensions
    const auto batch_size = input.size(0);
    const auto dim1 = input.size(1);
    const auto dim2 = input.size(2);
    
    // Only support reduction along dimension 1 for now
    TORCH_CHECK(dim == 1, "Only reduction along dimension 1 is supported");
    
    // Create output tensor
    auto output = torch::empty({batch_size, dim2}, input.options());
    
    // Configure kernel launch parameters
    const int warp_size = 32;
    const int warps_per_block = 8;
    const dim3 threads(warp_size, warps_per_block);
    const int blocks_x = (dim2 + warps_per_block - 1) / warps_per_block;
    const dim3 blocks(blocks_x, 1, batch_size);
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "product_reduction_cuda", ([&] {
        product_reduction_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            dim1,
            dim2
        );
    }));
    
    return output;
}
"""

cpp_source = """
#include <torch/extension.h>

torch::Tensor product_reduction_cuda(torch::Tensor input, int dim);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("product_reduction", &product_reduction_cuda, "Product reduction along a dimension (CUDA)");
}
"""

class ModelNew(nn.Module):
    """
    Optimized model that performs product reduction over a dimension.
    """
    def __init__(self, dim: int):
        """
        Initializes the model with the dimension to reduce over.

        Args:
            dim (int): Dimension to reduce over.
        """
        super(ModelNew, self).__init__()
        self.dim = dim
        self.product_cuda = None
        
        # Try to compile the CUDA extension
        try:
            self.product_cuda = load_inline(
                name="product_cuda",
                cpp_sources=cpp_source,
                cuda_sources=cuda_source,
                functions=["product_reduction"],
                with_cuda=True,
                verbose=False
            )
        except Exception as e:
            print(f"CUDA compilation failed: {e}")
            self.product_cuda = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs product reduction over the specified dimension using custom CUDA kernel.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor with product reduction applied.
        """
        # Use the custom CUDA kernel for product reduction when applicable
        if self.product_cuda is not None and x.is_cuda and self.dim == 1 and x.dim() == 3:
            return self.product_cuda.product_reduction(x, self.dim)
        else:
            # Fall back to PyTorch implementation
            return torch.prod(x, dim=self.dim)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim1 = 256
dim2 = 256
reduction_dim = 1

def get_inputs():
    x = torch.randn(batch_size, dim1, dim2)
    return [x]

def get_init_inputs():
    return [reduction_dim]