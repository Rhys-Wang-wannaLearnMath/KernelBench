import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized softplus CUDA kernel using float4 and aggressive thread coarsening
__global__ void softplus_kernel_optimized(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const int size_vec4) {
    
    // Each thread processes 16 elements (4 float4 vectors)
    const int elements_per_thread = 4; // in float4 units (= 16 float elements)
    const int idx_base = (blockIdx.x * blockDim.x + threadIdx.x) * elements_per_thread;
    const int stride = blockDim.x * gridDim.x * elements_per_thread;
    const float threshold = 20.0f;
    
    for (int i = idx_base; i < size_vec4; i += stride) {
        // Process 4 float4 vectors (16 elements total) per thread
        for (int j = 0; j < elements_per_thread && i + j < size_vec4; j++) {
            // Load float4 vector
            const float4 x4 = input[i + j];
            float4 result;
            
            // Process each component with minimal branching using ternary operators
            // x component
            result.x = (x4.x > threshold) ? x4.x : 
                      ((x4.x > 0.0f) ? (x4.x + __logf(1.0f + __expf(-x4.x))) : 
                                       __logf(1.0f + __expf(x4.x)));
            
            // y component
            result.y = (x4.y > threshold) ? x4.y : 
                      ((x4.y > 0.0f) ? (x4.y + __logf(1.0f + __expf(-x4.y))) : 
                                       __logf(1.0f + __expf(x4.y)));
            
            // z component
            result.z = (x4.z > threshold) ? x4.z : 
                      ((x4.z > 0.0f) ? (x4.z + __logf(1.0f + __expf(-x4.z))) : 
                                       __logf(1.0f + __expf(x4.z)));
            
            // w component
            result.w = (x4.w > threshold) ? x4.w : 
                      ((x4.w > 0.0f) ? (x4.w + __logf(1.0f + __expf(-x4.w))) : 
                                       __logf(1.0f + __expf(x4.w)));
            
            // Store result
            output[i + j] = result;
        }
    }
}

// Standard kernel for handling non-float types or sizes not divisible by 4
template <typename scalar_t>
__global__ void softplus_kernel_generic(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const scalar_t threshold = 20.0f;
    
    for (int i = idx; i < size; i += stride) {
        const scalar_t x = input[i];
        
        // Optimized branching using ternary operators
        output[i] = (x > threshold) ? x : 
                   ((x > 0.0f) ? (x + __logf(1.0f + __expf(-x))) : 
                                 __logf(1.0f + __expf(x)));
    }
}

torch::Tensor softplus_cuda_forward(torch::Tensor input) {
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Optimize thread configuration
    const int threads = 128; // Reduced from 256 to potentially increase occupancy
    
    // Use vectorized version for float tensors with size divisible by 4
    if (input.scalar_type() == torch::ScalarType::Float && size % 4 == 0) {
        const int size_vec4 = size / 4;
        
        // Each thread processes 16 elements (4 float4 vectors)
        // Calculate grid size accordingly
        const int elements_per_thread = 4; // in float4 units
        const int max_blocks = 1024;
        const int blocks = min(max_blocks, (size_vec4 + threads * elements_per_thread - 1) / (threads * elements_per_thread));
        
        softplus_kernel_optimized<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            size_vec4
        );
    } else {
        // Use standard version for other cases
        const int max_blocks = 1024;
        const int blocks = min(max_blocks, (size + threads - 1) / threads);
        
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "softplus_cuda_forward", ([&] {
            softplus_kernel_generic<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                size
            );
        }));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softplus_cuda_forward, "Softplus forward (CUDA)");
}
"""

class ModelNew(nn.Module):
    """
    Simple model that performs a Softplus activation with optimized CUDA kernel.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
        self.use_cuda_kernel = False
        
        # Try to compile the CUDA kernel
        try:
            self.softplus_cuda = load_inline(
                name="softplus_cuda",
                cpp_sources="",
                cuda_sources=cuda_source,
                functions=["forward"],
                with_cuda=True,
                extra_cuda_cflags=["-O3", "--use_fast_math"]
            )
            self.use_cuda_kernel = torch.cuda.is_available()
        except Exception as e:
            print(f"CUDA compilation failed: {e}")
            self.use_cuda_kernel = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Softplus activation to the input tensor.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Softplus applied, same shape as input.
        """
        if self.use_cuda_kernel and x.is_cuda:
            return self.softplus_cuda.forward(x)
        else:
            # Fallback to PyTorch implementation
            return torch.nn.functional.softplus(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed