import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel for product reduction
template <typename scalar_t>
__global__ void product_reduction_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    // Each thread handles one element in the output tensor
    const int batch_idx = blockIdx.x;
    const int col_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && col_idx < dim2) {
        // First check for zeros - early termination
        bool has_zero = false;
        
        // Check for zeros in chunks of 8 for efficiency
        int i = 0;
        for (; i <= dim1 - 8; i += 8) {
            if (input[batch_idx * dim1 * dim2 + (i+0) * dim2 + col_idx] == 0.0 ||
                input[batch_idx * dim1 * dim2 + (i+1) * dim2 + col_idx] == 0.0 ||
                input[batch_idx * dim1 * dim2 + (i+2) * dim2 + col_idx] == 0.0 ||
                input[batch_idx * dim1 * dim2 + (i+3) * dim2 + col_idx] == 0.0 ||
                input[batch_idx * dim1 * dim2 + (i+4) * dim2 + col_idx] == 0.0 ||
                input[batch_idx * dim1 * dim2 + (i+5) * dim2 + col_idx] == 0.0 ||
                input[batch_idx * dim1 * dim2 + (i+6) * dim2 + col_idx] == 0.0 ||
                input[batch_idx * dim1 * dim2 + (i+7) * dim2 + col_idx] == 0.0) {
                has_zero = true;
                break;
            }
        }
        
        // Check remaining elements individually
        if (!has_zero) {
            for (; i < dim1; ++i) {
                if (input[batch_idx * dim1 * dim2 + i * dim2 + col_idx] == 0.0) {
                    has_zero = true;
                    break;
                }
            }
        }
        
        if (has_zero) {
            // Early termination - result is zero
            output[batch_idx * dim2 + col_idx] = 0.0;
            return;
        }
        
        // No zeros found, compute the product with multiple accumulators
        scalar_t product1 = 1.0;
        scalar_t product2 = 1.0;
        scalar_t product3 = 1.0;
        scalar_t product4 = 1.0;
        
        // Process in chunks of 16 with 4 accumulators for better ILP
        i = 0;
        for (; i <= dim1 - 16; i += 16) {
            product1 *= input[batch_idx * dim1 * dim2 + (i+0) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+1) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+2) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+3) * dim2 + col_idx];
                       
            product2 *= input[batch_idx * dim1 * dim2 + (i+4) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+5) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+6) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+7) * dim2 + col_idx];
                       
            product3 *= input[batch_idx * dim1 * dim2 + (i+8) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+9) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+10) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+11) * dim2 + col_idx];
                       
            product4 *= input[batch_idx * dim1 * dim2 + (i+12) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+13) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+14) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+15) * dim2 + col_idx];
        }
        
        // Handle remaining elements in groups of 4
        for (; i <= dim1 - 4; i += 4) {
            product1 *= input[batch_idx * dim1 * dim2 + (i+0) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+1) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+2) * dim2 + col_idx] * 
                       input[batch_idx * dim1 * dim2 + (i+3) * dim2 + col_idx];
        }
        
        // Handle remaining elements individually
        for (; i < dim1; ++i) {
            product1 *= input[batch_idx * dim1 * dim2 + i * dim2 + col_idx];
        }
        
        // Combine all accumulators and write result
        output[batch_idx * dim2 + col_idx] = product1 * product2 * product3 * product4;
    }
}

// Log-sum-exp kernel for numerical stability with large reduction dimensions
template <typename scalar_t>
__global__ void product_reduction_logsum_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int dim1,
    const int dim2) {
    
    const int batch_idx = blockIdx.x;
    const int col_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size && col_idx < dim2) {
        // First check for zeros - early termination
        bool has_zero = false;
        for (int i = 0; i < dim1; ++i) {
            if (input[batch_idx * dim1 * dim2 + i * dim2 + col_idx] == 0.0) {
                has_zero = true;
                break;
            }
        }
        
        if (has_zero) {
            // Early termination - result is zero
            output[batch_idx * dim2 + col_idx] = 0.0;
            return;
        }
        
        // Use log-sum-exp approach for numerical stability
        scalar_t log_sum = 0.0;
        int sign_count = 0;
        
        for (int i = 0; i < dim1; ++i) {
            scalar_t val = input[batch_idx * dim1 * dim2 + i * dim2 + col_idx];
            log_sum += log(fabs(val));
            sign_count += (val < 0) ? 1 : 0;
        }
        
        // Compute final result with correct sign
        scalar_t result = exp(log_sum);
        if (sign_count % 2 == 1) {
            result = -result;
        }
        
        output[batch_idx * dim2 + col_idx] = result;
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
    
    // Choose kernel based on reduction size
    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "product_reduction_cuda", ([&] {
        if (dim1 > 512) {
            // For large reduction dimensions, use log-sum-exp for numerical stability
            const int threads = 256;
            const int blocks_y = (dim2 + threads - 1) / threads;
            const dim3 blocks(batch_size, blocks_y);
            
            product_reduction_logsum_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim1,
                dim2
            );
        } else {
            // For smaller reduction dimensions, use standard kernel
            const int threads = 256;
            const int blocks_y = (dim2 + threads - 1) / threads;
            const dim3 blocks(batch_size, blocks_y);
            
            product_reduction_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                batch_size,
                dim1,
                dim2
            );
        }
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