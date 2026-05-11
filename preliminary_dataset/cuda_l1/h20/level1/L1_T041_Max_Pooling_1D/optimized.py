import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
import math

# Define CUDA kernel for 1D max pooling
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Specialized kernel for benchmark parameters: kernel_size=4, stride=2, padding=2, dilation=3
template <typename scalar_t>
__global__ void max_pool1d_benchmark_kernel(
    const scalar_t* __restrict__ input,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int channels,
    const int input_length,
    const int output_length) {
    
    // Get position
    const int b = blockIdx.x;  // batch
    const int c = blockIdx.y;  // channel
    const int tid = threadIdx.x;
    const int stride = blockDim.x;
    
    // Constants for the benchmark case
    const int kernel_size = 4;
    const int step_stride = 2;
    const int padding = 2;
    const int dilation = 3;
    
    // Calculate input and output offsets
    const int input_offset = b * channels * input_length + c * input_length;
    const int output_offset = b * channels * output_length + c * output_length;
    
    // Process multiple output elements per thread if needed
    for (int o_idx = tid; o_idx < output_length; o_idx += stride) {
        // Calculate the start position in the input for this output element
        const int i_start = o_idx * step_stride - padding;
        
        // Initialize with lowest possible value
        scalar_t max_val = -std::numeric_limits<scalar_t>::infinity();
        
        // Unrolled loop for kernel_size=4, dilation=3
        // Position 0
        if (i_start >= 0 && i_start < input_length) {
            max_val = input[input_offset + i_start];
        }
        
        // Position 1
        if (i_start + dilation >= 0 && i_start + dilation < input_length) {
            max_val = max(max_val, input[input_offset + i_start + dilation]);
        }
        
        // Position 2
        if (i_start + 2 * dilation >= 0 && i_start + 2 * dilation < input_length) {
            max_val = max(max_val, input[input_offset + i_start + 2 * dilation]);
        }
        
        // Position 3
        if (i_start + 3 * dilation >= 0 && i_start + 3 * dilation < input_length) {
            max_val = max(max_val, input[input_offset + i_start + 3 * dilation]);
        }
        
        // Write output
        output[output_offset + o_idx] = max_val;
    }
}

torch::Tensor max_pool1d_cuda_forward(torch::Tensor input) {
    // Get dimensions
    const int batch_size = input.size(0);
    const int channels = input.size(1);
    const int input_length = input.size(2);
    
    // Benchmark parameters
    const int kernel_size = 4;
    const int stride = 2;
    const int padding = 2;
    const int dilation = 3;
    
    // Calculate output size
    const int output_length = ((input_length + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1;
    
    // Create output tensor
    auto output = torch::empty({batch_size, channels, output_length}, input.options());
    
    // Determine optimal thread block size
    const int threads_per_block = 128;  // Can be tuned for best performance
    
    // Calculate grid dimensions
    const dim3 blocks(batch_size, channels);
    const dim3 threads(threads_per_block);
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "max_pool1d_benchmark_kernel", ([&] {
        max_pool1d_benchmark_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            channels,
            input_length,
            output_length);
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &max_pool1d_cuda_forward, "MaxPool1D benchmark forward (CUDA)");
}
"""

# Try to load the custom CUDA extension
try:
    max_pool1d_cuda = load_inline(
        name="max_pool1d_cuda",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["forward"],
        with_cuda=True,
        extra_cuda_cflags=["-O3"]
    )
    CUDA_EXTENSION_AVAILABLE = True
except Exception as e:
    print(f"Failed to load CUDA extension: {e}")
    CUDA_EXTENSION_AVAILABLE = False

class ModelNew(nn.Module):
    """
    Optimized implementation of Max Pooling 1D with custom CUDA kernels.
    
    Args:
        kernel_size (int): Size of the window to take a max over.
        stride (int, optional): Stride of the window. Defaults to None (same as kernel_size).
        padding (int, optional): Implicit zero padding to be added on both sides. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        return_indices (bool, optional): Whether to return the indices of the maximum values. Defaults to False.
    """
    def __init__(self, kernel_size: int, stride: int = None, padding: int = 0, dilation: int = 1, return_indices: bool = False):
        super(ModelNew, self).__init__()
        
        # Cache parameters
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        
        # Check if we're using the benchmark parameters
        self.is_benchmark = (kernel_size == 4 and self.stride == 2 and 
                            padding == 2 and dilation == 3 and not return_indices)
        
        # Select the optimal forward implementation at initialization time
        if return_indices:
            # Must use nn.MaxPool1d for indices
            self.maxpool = nn.MaxPool1d(
                kernel_size=kernel_size,
                stride=self.stride,
                padding=padding,
                dilation=dilation,
                return_indices=True
            )
            # Replace the forward method with the specialized implementation
            self.forward = self._forward_with_indices
        elif self.is_benchmark and CUDA_EXTENSION_AVAILABLE:
            # Use our optimized CUDA kernel for the benchmark case
            self.forward = self._forward_cuda
        else:
            # Ultra-optimized path with hardcoded parameters for benchmark case
            if self.is_benchmark:
                # Using a direct lambda to eliminate all overhead
                self.forward = lambda x: F.max_pool1d(x, 4, 2, 2, 3)
            else:
                # General case with cached parameters
                self.forward = self._forward_general
    
    def _forward_with_indices(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for indices case."""
        return self.maxpool(x)
    
    def _forward_cuda(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass using custom CUDA kernel."""
        # Make sure input is contiguous for our kernel
        if not x.is_cuda:
            x = x.cuda()
        if not x.is_contiguous():
            x = x.contiguous()
        return max_pool1d_cuda.forward(x)
    
    def _forward_general(self, x: torch.Tensor) -> torch.Tensor:
        """General case forward pass."""
        return F.max_pool1d(x, self.kernel_size, self.stride, self.padding, self.dilation)
    
    # This forward method will be replaced at initialization time
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies Max Pooling 1D to the input tensor.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_features, sequence_length).
            
        Returns:
            torch.Tensor: Output tensor with Max Pooling 1D applied.
        """
        pass

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
features = 64
sequence_length = 128
kernel_size = 4
stride = 2
padding = 2
dilation = 3
return_indices = False

def get_inputs():
    x = torch.randn(batch_size, features, sequence_length)
    return [x]

def get_init_inputs():
    return [kernel_size, stride, padding, dilation, return_indices]