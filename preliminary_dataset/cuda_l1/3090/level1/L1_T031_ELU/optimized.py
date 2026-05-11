import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline, CUDA_HOME
import os

# Check if CUDA is available
has_cuda = torch.cuda.is_available() and CUDA_HOME is not None

if has_cuda:
    # Define CUDA kernel with optimizations
    cuda_source = """
    #include <torch/extension.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    
    __device__ __forceinline__ float elu_op(float x, float alpha) {
        return x > 0.0f ? x : alpha * (__expf(x) - 1.0f);
    }
    
    // Optimized ELU kernel with float4 vectorization
    __global__ void elu_cuda_kernel_optimized(
        const float* __restrict__ input,
        float* __restrict__ output,
        const float alpha,
        const int size) {
        
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        
        // Process elements using float4 for vectorized memory access
        for (int i = tid; i < size / 4; i += stride) {
            const int idx = i * 4;
            
            // Load 4 elements as float4 for coalesced access
            float4 data = *reinterpret_cast<const float4*>(&input[idx]);
            
            // Apply ELU operation to all 4 elements
            data.x = elu_op(data.x, alpha);
            data.y = elu_op(data.y, alpha);
            data.z = elu_op(data.z, alpha);
            data.w = elu_op(data.w, alpha);
            
            // Store results as float4
            *reinterpret_cast<float4*>(&output[idx]) = data;
        }
        
        // Handle remaining elements (0-3 elements)
        const int remainder_start = (size / 4) * 4;
        for (int idx = remainder_start + tid; idx < size; idx += stride) {
            output[idx] = elu_op(input[idx], alpha);
        }
    }
    
    torch::Tensor elu_cuda_forward(torch::Tensor input, float alpha) {
        auto output = torch::empty_like(input);
        const int size = input.numel();
        
        if (size == 0) return output;
        
        // Use 256 threads per block - good balance for memory-bound operations
        const int threads = 256;
        
        // Calculate optimal grid size based on tensor size
        // For memory-bound operations, we want enough blocks to keep the GPU busy
        // but not too many to cause excessive scheduling overhead
        int device_id = input.get_device();
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        
        // Calculate blocks based on SM count and occupancy goals
        int num_sms = prop.multiProcessorCount;
        int blocks_per_sm = 2;  // Aim for 2 blocks per SM for good occupancy
        int target_blocks = num_sms * blocks_per_sm;
        
        // Ensure we have at least enough blocks to cover the data
        int min_blocks_needed = (size + threads * 4 - 1) / (threads * 4);
        int blocks = max(min_blocks_needed, min(target_blocks, 1024));
        
        // Launch optimized kernel
        elu_cuda_kernel_optimized<<<blocks, threads>>>(
            input.data_ptr<float>(),
            output.data_ptr<float>(),
            alpha,
            size);
        
        return output;
    }
    """
    
    cpp_source = """
    #include <torch/extension.h>
    
    torch::Tensor elu_cuda_forward(torch::Tensor input, float alpha);
    
    torch::Tensor elu_forward(torch::Tensor input, float alpha) {
        return elu_cuda_forward(input, alpha);
    }
    
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("forward", &elu_forward, "ELU forward (CUDA)");
    }
    """
    
    # JIT compile the CUDA extension with optimization flags
    try:
        elu_cuda = load_inline(
            name="elu_cuda_optimized",
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            functions=["forward"],
            verbose=False,
            extra_cuda_cflags=["--use_fast_math", "-O3"]
        )
        has_cuda_extension = True
    except Exception as e:
        print(f"Failed to load CUDA extension: {e}")
        has_cuda_extension = False
else:
    has_cuda_extension = False

class ModelNew(nn.Module):
    """
    Optimized model that performs an ELU activation.
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initializes the ELU model.

        Args:
            alpha (float, optional): The alpha parameter for the ELU function. Defaults to 1.0.
        """
        super(ModelNew, self).__init__()
        self.alpha = alpha
        self.use_cuda_kernel = has_cuda and has_cuda_extension
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies ELU activation to the input tensor using an optimized implementation.

        Args:
            x (torch.Tensor): Input tensor of any shape.

        Returns:
            torch.Tensor: Output tensor with ELU applied, same shape as input.
        """
        # If CUDA is available and extension loaded successfully, use optimized kernel
        if self.use_cuda_kernel and x.is_cuda and x.dtype == torch.float32:
            # Ensure tensor is contiguous for optimal performance
            if not x.is_contiguous():
                x = x.contiguous()
                
            try:
                return elu_cuda.forward(x, self.alpha)
            except Exception:
                # Fallback to PyTorch's implementation if kernel fails
                return F.elu(x, alpha=self.alpha)
        else:
            # Use PyTorch's native implementation
            return F.elu(x, alpha=self.alpha)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
dim = 16384

def get_inputs():
    x = torch.randn(batch_size, dim)
    return [x]

def get_init_inputs():
    return [1.0]  # Provide alpha value for initialization