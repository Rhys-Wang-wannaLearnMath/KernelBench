import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline
import os

# Define CUDA kernel for transposed convolution with asymmetric (3,5) kernel
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

// Optimized CUDA kernel for transposed convolution with (3,5) kernel
template <typename scalar_t>
__global__ void transposed_conv2d_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ output,
    const int batch_size,
    const int in_channels,
    const int out_channels,
    const int in_height,
    const int in_width,
    const int out_height,
    const int out_width,
    const int stride,
    const int padding,
    const int output_padding,
    const int groups) {
    
    // Constants for kernel size - hardcoded for (3,5)
    const int kernel_h = 3;
    const int kernel_w = 5;
    
    // Shared memory for weights
    extern __shared__ char shared_memory[];
    scalar_t* shared_weights = reinterpret_cast<scalar_t*>(shared_memory);
    
    // Block dimensions
    const int BLOCK_W = blockDim.x;  // 16
    const int BLOCK_H = blockDim.y;  // 16
    
    // Calculate output pixel positions
    const int out_x = blockIdx.x * BLOCK_W + threadIdx.x;
    const int out_y = blockIdx.y * BLOCK_H + threadIdx.y;
    const int out_c = blockIdx.z % out_channels;
    const int b = blockIdx.z / out_channels;
    
    // Calculate group information
    const int in_c_per_group = in_channels / groups;
    const int out_c_per_group = out_channels / groups;
    const int group = out_c / out_c_per_group;
    
    // Load weights into shared memory
    const int thread_idx = threadIdx.y * BLOCK_W + threadIdx.x;
    const int total_threads = BLOCK_W * BLOCK_H;
    const int weights_per_channel = kernel_h * kernel_w;
    const int total_weights = in_c_per_group * weights_per_channel;
    
    // Collaborative loading of weights into shared memory
    for (int i = thread_idx; i < total_weights; i += total_threads) {
        const int ic = i / weights_per_channel;
        const int k_idx = i % weights_per_channel;
        const int kh = k_idx / kernel_w;
        const int kw = k_idx % kernel_w;
        
        // Load weight with reversed indices for transposed conv
        shared_weights[i] = weight[
            ((out_c * in_c_per_group + ic) * kernel_h + (kernel_h - 1 - kh)) * kernel_w + (kernel_w - 1 - kw)
        ];
    }
    
    // Ensure all weights are loaded
    __syncthreads();
    
    // Skip if out of bounds
    if (out_x >= out_width || out_y >= out_height || b >= batch_size)
        return;
    
    // Initialize accumulator
    scalar_t value = 0;
    
    // Precompute output index to reduce redundant calculations
    const int out_idx = ((b * out_channels + out_c) * out_height + out_y) * out_width + out_x;
    
    // Iterate over input channels in this group
    for (int ic = 0; ic < in_c_per_group; ++ic) {
        const int in_c = group * in_c_per_group + ic;
        const int in_batch_ch_offset = (b * in_channels + in_c) * in_height;
        
        // Iterate over kernel - fully unrolled for (3,5) kernel
        #pragma unroll
        for (int kh = 0; kh < kernel_h; ++kh) {
            const int in_y = (out_y + padding - kh) / stride;
            const bool valid_h = in_y >= 0 && in_y < in_height && (out_y + padding - kh) % stride == 0;
            
            if (valid_h) {
                const int in_row_offset = in_batch_ch_offset + in_y * in_width;
                
                #pragma unroll
                for (int kw = 0; kw < kernel_w; ++kw) {
                    const int in_x = (out_x + padding - kw) / stride;
                    
                    // Check if the input position is valid and contributes to this output
                    if (in_x >= 0 && in_x < in_width && (out_x + padding - kw) % stride == 0) {
                        // Get input value
                        const scalar_t in_val = input[in_row_offset + in_x];
                        
                        // Get weight value from shared memory
                        const scalar_t w_val = shared_weights[
                            ic * weights_per_channel + kh * kernel_w + kw
                        ];
                        
                        // Accumulate
                        value += in_val * w_val;
                    }
                }
            }
        }
    }
    
    // Write output
    output[out_idx] = value;
}

// C++ interface
torch::Tensor transposed_conv2d_cuda(
    torch::Tensor input,
    torch::Tensor weight,
    int stride,
    int padding,
    int output_padding,
    int groups) {
    
    // Get dimensions
    const int batch_size = input.size(0);
    const int in_channels = input.size(1);
    const int in_height = input.size(2);
    const int in_width = input.size(3);
    
    const int out_channels = weight.size(0);
    const int kernel_h = weight.size(2);
    const int kernel_w = weight.size(3);
    
    // Calculate output dimensions
    const int out_height = (in_height - 1) * stride - 2 * padding + kernel_h + output_padding;
    const int out_width = (in_width - 1) * stride - 2 * padding + kernel_w + output_padding;
    
    // Create output tensor
    auto output = torch::zeros({batch_size, out_channels, out_height, out_width},
                              input.options());
    
    // Calculate grid and block dimensions
    const int BLOCK_W = 16;
    const int BLOCK_H = 16;
    const int grid_x = (out_width + BLOCK_W - 1) / BLOCK_W;
    const int grid_y = (out_height + BLOCK_H - 1) / BLOCK_H;
    const int grid_z = batch_size * out_channels;
    
    const dim3 grid(grid_x, grid_y, grid_z);
    const dim3 block(BLOCK_W, BLOCK_H);
    
    // Calculate shared memory size
    const int in_c_per_group = in_channels / groups;
    const int shared_mem_size = in_c_per_group * kernel_h * kernel_w * sizeof(float);
    
    // Launch kernel
    AT_DISPATCH_FLOATING_TYPES(input.type(), "transposed_conv2d_cuda", ([&] {
        transposed_conv2d_kernel<scalar_t><<<grid, block, shared_mem_size>>>(
            input.data_ptr<scalar_t>(),
            weight.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            batch_size,
            in_channels,
            out_channels,
            in_height,
            in_width,
            out_height,
            out_width,
            stride,
            padding,
            output_padding,
            groups
        );
    }));
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("transposed_conv2d", &transposed_conv2d_cuda, "Transposed Convolution 2D CUDA");
}
"""

class ModelNew(nn.Module):
    """
    Performs a transposed 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (height, width).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        output_padding (int, optional): Additional size added to one side of the output shape. Defaults to 0.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False):
        super(ModelNew, self).__init__()
        
        # Create standard PyTorch ConvTranspose2d layer as fallback
        self.conv_transpose2d = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding,
            output_padding=output_padding, groups=groups, bias=bias
        )
        
        # Store parameters for custom kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.output_padding = output_padding
        self.groups = groups
        
        # Try to load the CUDA extension
        try:
            self.transposed_conv_cuda = load_inline(
                name=f"transposed_conv_cuda_{os.getpid()}",  # Add PID to avoid name conflicts
                cpp_sources="",
                cuda_sources=cuda_source,
                functions=["transposed_conv2d"],
                verbose=True,
                with_cuda=True
            )
            self.cuda_extension_loaded = True
        except Exception as e:
            print(f"Failed to load CUDA extension: {e}")
            self.cuda_extension_loaded = False
        
        # Flag to use custom CUDA kernel
        self.use_custom_kernel = self.cuda_extension_loaded and torch.cuda.is_available()
        
        # Enable cuDNN optimizations for fallback
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            
            # Convert weights to channels_last format for better performance in fallback
            if not self.use_custom_kernel:
                self.conv_transpose2d.weight.data = self.conv_transpose2d.weight.data.to(memory_format=torch.channels_last)
        
        # Warmup flag
        self.warmed_up = False

    def forward(self, x):
        """
        Performs the transposed 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Use custom CUDA kernel if available and input is on CUDA
        if self.use_custom_kernel and x.is_cuda and self.kernel_size == (3, 5):
            # Ensure input is contiguous
            if not x.is_contiguous():
                x = x.contiguous()
            
            # Perform warmup pass if needed
            if not self.warmed_up:
                with torch.no_grad():
                    # Do a forward pass with the custom kernel to warm up
                    _ = self.transposed_conv_cuda.transposed_conv2d(
                        x.clone(),
                        self.conv_transpose2d.weight,
                        self.stride,
                        self.padding,
                        self.output_padding,
                        self.groups
                    )
                    torch.cuda.synchronize()
                    self.warmed_up = True
            
            # Use custom CUDA kernel
            return self.transposed_conv_cuda.transposed_conv2d(
                x,
                self.conv_transpose2d.weight,
                self.stride,
                self.padding,
                self.output_padding,
                self.groups
            )
        else:
            # Fallback to PyTorch implementation
            
            # Convert to channels_last memory format if on CUDA for better performance
            if x.is_cuda:
                x = x.to(memory_format=torch.channels_last)
                
                # Ensure x is contiguous in the channels_last memory format
                if not x.is_contiguous(memory_format=torch.channels_last):
                    x = x.contiguous(memory_format=torch.channels_last)
            elif not x.is_contiguous():
                x = x.contiguous()
            
            # Perform warmup pass if needed
            if not self.warmed_up and x.is_cuda:
                with torch.no_grad():
                    # Do a forward pass to warm up cuDNN
                    _ = self.conv_transpose2d(x.clone())
                    torch.cuda.synchronize()
                    self.warmed_up = True
            
            # Use PyTorch's implementation
            return self.conv_transpose2d(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
width = 128
height = 128

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization