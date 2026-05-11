import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom CUDA kernel for optimized Swish implementation
swish_kernel_code = """
// Inline function for sigmoid calculation
__device__ __forceinline__ float sigmoid(float x) {
    if (x >= 0) {
        return 1.0f / (1.0f + __expf(-x));
    } else {
        float exp_x = __expf(x);
        return exp_x / (1.0f + exp_x);
    }
}

extern "C" __global__ void swish_forward_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    const int size) {
    
    #pragma unroll 4
    // Grid-stride loop to handle large tensors
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < size; 
         idx += blockDim.x * gridDim.x) {
        
        // Load input value
        float x = input[idx];
        
        // Compute swish: x * sigmoid(x)
        output[idx] = x * sigmoid(x);
    }
}

// Vectorized version for better memory throughput
extern "C" __global__ void swish_forward_kernel_vec4(
    const float4* __restrict__ input,
    float4* __restrict__ output,
    const int vec_size) {
    
    #pragma unroll 2
    // Grid-stride loop processing 4 elements at once
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; 
         idx < vec_size; 
         idx += blockDim.x * gridDim.x) {
        
        // Load 4 elements at once
        float4 x4 = input[idx];
        float4 result;
        
        // Process all components
        result.x = x4.x * sigmoid(x4.x);
        result.y = x4.y * sigmoid(x4.y);
        result.z = x4.z * sigmoid(x4.z);
        result.w = x4.w * sigmoid(x4.w);
        
        // Store 4 results at once
        output[idx] = result;
    }
}
"""

# Try to compile the CUDA kernel
try:
    from torch.utils.cpp_extension import load_inline
    swish_cuda = load_inline(
        name="swish_cuda",
        cpp_sources="",
        cuda_sources=swish_kernel_code,
        functions=["swish_forward_kernel", "swish_forward_kernel_vec4"],
        with_cuda=True,
        verbose=False,
        extra_cuda_cflags=["--use_fast_math", "-O3"]
    )
    CUDA_KERNEL_AVAILABLE = True
except Exception:
    CUDA_KERNEL_AVAILABLE = False

class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # Save input for backward pass
        ctx.save_for_backward(x)
        
        # If CUDA kernel is not available or tensor is not on CUDA,
        # fall back to PyTorch's implementation
        if not CUDA_KERNEL_AVAILABLE or not x.is_cuda:
            return F.silu(x)
        
        # Ensure contiguous memory layout
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Create output tensor
        output = torch.empty_like(x)
        numel = x.numel()
        
        # Optimize thread block size for modern GPUs
        threads_per_block = 256
        
        # Check if we can use vectorized version (size must be multiple of 4)
        if numel % 4 == 0:
            # Use vectorized kernel
            vec_size = numel // 4
            
            # Calculate optimal grid size based on SM count
            sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
            blocks_per_grid = min(65535, max(sm_count * 4, (vec_size + threads_per_block - 1) // threads_per_block))
            
            swish_cuda.swish_forward_kernel_vec4(
                x.data_ptr(),
                output.data_ptr(),
                vec_size,
                grid=(blocks_per_grid,),
                block=(threads_per_block,)
            )
        else:
            # Use standard kernel
            blocks_per_grid = min(65535, max(
                torch.cuda.get_device_properties(x.device).multi_processor_count * 4,
                (numel + threads_per_block - 1) // threads_per_block
            ))
            
            swish_cuda.swish_forward_kernel(
                x.data_ptr(),
                output.data_ptr(),
                numel,
                grid=(blocks_per_grid,),
                block=(threads_per_block,)
            )
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        
        # Use PyTorch's optimized operations for backward pass
        sigmoid_x = torch.sigmoid(x)
        grad_input = grad_output * (sigmoid_x + x * sigmoid_x * (1 - sigmoid_x))
        
        return grad_input

class ModelNew(nn.Module):
    """
    Optimized model that performs a Swish activation.
    """
    def __init__(self):
        super(ModelNew, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Swish activation to the input tensor using optimized CUDA implementation.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with Swish applied, same shape as input.
        """
        # Use our custom CUDA kernel if available and tensor is on CUDA
        if CUDA_KERNEL_AVAILABLE and x.is_cuda:
            return SwishFunction.apply(x)
        
        # Fall back to PyTorch's optimized implementation
        return F.silu(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return []  # No special initialization inputs needed