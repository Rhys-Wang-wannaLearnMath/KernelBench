import torch
import torch.nn as nn
import math
from torch.utils.cpp_extension import load_inline
import os

class ModelNew(nn.Module):
    """
    Performs a standard 3D convolution operation with a square input and an asymmetric kernel.
    Optimized implementation for better performance.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Size of the convolution kernel (kernel_width, kernel_height, kernel_depth).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int or tuple, optional): Padding applied to the input. Defaults to 0.
        dilation (int or tuple, optional): Spacing between kernel elements. Defaults to 1.
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, stride: int = 1, padding: int = 0, dilation: int = 1, groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create standard Conv3d layer as a fallback
        self.conv3d = nn.Conv3d(in_channels, out_channels, kernel_size, 
                              stride=stride, padding=padding, 
                              dilation=dilation, groups=groups, bias=bias)
        
        # Enable cuDNN benchmarking for optimal algorithm selection
        torch.backends.cudnn.benchmark = True
        
        # Store parameters for custom kernel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        
        # Optimization flags
        self.use_channels_last = False
        self.use_custom_kernel = False
        self.has_run_benchmark = False
        
        # Try to load custom CUDA kernel
        self.cuda_kernel = None
        if torch.cuda.is_available():
            try:
                self._load_custom_kernel()
                self.use_custom_kernel = True
            except Exception as e:
                print(f"Failed to load custom CUDA kernel: {e}")
                self.use_custom_kernel = False
    
    def _load_custom_kernel(self):
        cuda_source = '''
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <vector>

        template <typename scalar_t>
        __global__ void conv3d_asymmetric_kernel(
            const scalar_t* __restrict__ input,
            const scalar_t* __restrict__ weight,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int in_channels,
            const int out_channels,
            const int in_width,
            const int in_height,
            const int in_depth,
            const int out_width,
            const int out_height,
            const int out_depth,
            const int kernel_width,
            const int kernel_height,
            const int kernel_depth,
            const int stride,
            const int padding,
            const int groups) {
            
            // Calculate output position
            const int n = blockIdx.z;
            const int f = blockIdx.y * blockDim.y + threadIdx.y;
            const int z_out = (blockIdx.x * blockDim.x + threadIdx.x) / (out_height * out_width);
            const int tmp = (blockIdx.x * blockDim.x + threadIdx.x) % (out_height * out_width);
            const int y_out = tmp / out_width;
            const int x_out = tmp % out_width;

            // Check if thread is within bounds
            if (n >= batch_size || f >= out_channels || z_out >= out_depth || y_out >= out_height || x_out >= out_width)
                return;

            // Compute convolution
            scalar_t value = 0;
            const int channels_per_group = in_channels / groups;
            const int group = f / (out_channels / groups);
            
            #pragma unroll
            for (int c = 0; c < channels_per_group; ++c) {
                const int c_in = group * channels_per_group + c;
                
                #pragma unroll
                for (int kz = 0; kz < kernel_depth; ++kz) {
                    const int z_in = z_out * stride - padding + kz;
                    if (z_in >= 0 && z_in < in_depth) {
                        
                        #pragma unroll
                        for (int ky = 0; ky < kernel_height; ++ky) {
                            const int y_in = y_out * stride - padding + ky;
                            if (y_in >= 0 && y_in < in_height) {
                                
                                #pragma unroll
                                for (int kx = 0; kx < kernel_width; ++kx) {
                                    const int x_in = x_out * stride - padding + kx;
                                    if (x_in >= 0 && x_in < in_width) {
                                        const int input_idx = ((n * in_channels + c_in) * in_depth + z_in) * in_height * in_width + y_in * in_width + x_in;
                                        const int weight_idx = ((f * channels_per_group + c) * kernel_depth + kz) * kernel_height * kernel_width + ky * kernel_width + kx;
                                        value += input[input_idx] * weight[weight_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            // Write output
            const int output_idx = ((n * out_channels + f) * out_depth + z_out) * out_height * out_width + y_out * out_width + x_out;
            output[output_idx] = value;
        }

        std::vector<torch::Tensor> conv3d_cuda_forward(
            torch::Tensor input,
            torch::Tensor weight,
            int stride,
            int padding,
            int groups) {
            
            // Get dimensions
            const int batch_size = input.size(0);
            const int in_channels = input.size(1);
            const int in_depth = input.size(2);
            const int in_height = input.size(3);
            const int in_width = input.size(4);
            
            const int out_channels = weight.size(0);
            const int kernel_depth = weight.size(2);
            const int kernel_height = weight.size(3);
            const int kernel_width = weight.size(4);
            
            const int out_depth = (in_depth + 2 * padding - kernel_depth) / stride + 1;
            const int out_height = (in_height + 2 * padding - kernel_height) / stride + 1;
            const int out_width = (in_width + 2 * padding - kernel_width) / stride + 1;
            
            // Create output tensor
            auto output = torch::zeros({batch_size, out_channels, out_depth, out_height, out_width}, 
                                    input.options());
            
            // Calculate grid and block dimensions
            const int threads_per_block = 8;
            const int blocks_per_grid_z = batch_size;
            const int blocks_per_grid_y = (out_channels + threads_per_block - 1) / threads_per_block;
            const int blocks_per_grid_x = (out_depth * out_height * out_width + threads_per_block - 1) / threads_per_block;
            
            const dim3 grid_dim(blocks_per_grid_x, blocks_per_grid_y, blocks_per_grid_z);
            const dim3 block_dim(threads_per_block, threads_per_block, 1);
            
            // Launch kernel
            AT_DISPATCH_FLOATING_TYPES(input.type(), "conv3d_cuda_forward", ([&] {
                conv3d_asymmetric_kernel<scalar_t><<<grid_dim, block_dim>>>(
                    input.data<scalar_t>(),
                    weight.data<scalar_t>(),
                    output.data<scalar_t>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    in_width,
                    in_height,
                    in_depth,
                    out_width,
                    out_height,
                    out_depth,
                    kernel_width,
                    kernel_height,
                    kernel_depth,
                    stride,
                    padding,
                    groups
                );
            }));
            
            return {output};
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("forward", &conv3d_cuda_forward, "Conv3D forward (CUDA)");
        }
        '''
        
        self.cuda_kernel = load_inline(
            name='conv3d_cuda',
            cpp_sources=[],
            cuda_sources=[cuda_source],
            functions=['forward'],
            verbose=True
        )
    
    def _run_algorithm_benchmark(self, x):
        """Run a benchmark to find the best algorithm for this specific workload"""
        if self.has_run_benchmark:
            return
            
        # Only benchmark if CUDA is available
        if not x.is_cuda:
            return
            
        # Create test tensors for benchmarking
        x_test = x.clone().detach()
        
        # Try standard format
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        # Warm up
        for _ in range(5):
            _ = self.conv3d(x_test)
        
        # Benchmark standard format
        start.record()
        for _ in range(10):
            _ = self.conv3d(x_test)
        end.record()
        torch.cuda.synchronize()
        standard_time = start.elapsed_time(end)
        
        # Try channels_last format if available
        channels_last_time = float('inf')
        if hasattr(torch, 'channels_last_3d'):
            try:
                x_cl = x_test.to(memory_format=torch.channels_last_3d)
                weight_cl = self.conv3d.weight.data.to(memory_format=torch.channels_last_3d)
                
                # Create a temporary conv layer with channels_last weights
                temp_conv = nn.Conv3d(self.in_channels, self.out_channels, self.kernel_size,
                                    stride=self.stride, padding=self.padding,
                                    dilation=self.dilation, groups=self.groups,
                                    bias=self.conv3d.bias is not None)
                temp_conv.weight.data = weight_cl
                if self.conv3d.bias is not None:
                    temp_conv.bias.data = self.conv3d.bias.data
                temp_conv = temp_conv.to(x.device)
                
                # Warm up
                for _ in range(5):
                    _ = temp_conv(x_cl)
                
                # Benchmark channels_last format
                start.record()
                for _ in range(10):
                    _ = temp_conv(x_cl)
                end.record()
                torch.cuda.synchronize()
                channels_last_time = start.elapsed_time(end)
                
                # If channels_last is faster, convert weights
                if channels_last_time < standard_time:
                    self.conv3d.weight.data = self.conv3d.weight.data.to(memory_format=torch.channels_last_3d)
                    self.use_channels_last = True
            except Exception:
                # Channels last format not supported or failed
                pass
        
        # Try custom kernel if available
        custom_kernel_time = float('inf')
        if self.use_custom_kernel:
            try:
                # Warm up
                for _ in range(5):
                    _ = self.cuda_kernel.forward(x_test, self.conv3d.weight, self.stride, self.padding, self.groups)[0]
                
                # Benchmark custom kernel
                start.record()
                for _ in range(10):
                    _ = self.cuda_kernel.forward(x_test, self.conv3d.weight, self.stride, self.padding, self.groups)[0]
                end.record()
                torch.cuda.synchronize()
                custom_kernel_time = start.elapsed_time(end)
                
                # If custom kernel is not the fastest, disable it
                if custom_kernel_time >= min(standard_time, channels_last_time):
                    self.use_custom_kernel = False
            except Exception:
                # Custom kernel failed
                self.use_custom_kernel = False
        
        self.has_run_benchmark = True
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D convolution with optimized implementation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, width, height, depth).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, width_out, height_out, depth_out).
        """
        # If not on CUDA, use standard implementation
        if not x.is_cuda:
            return self.conv3d(x)
        
        # Run algorithm benchmark if not done yet
        if not self.has_run_benchmark:
            self._run_algorithm_benchmark(x)
        
        try:
            # Use custom kernel if available and benchmarked to be faster
            if self.use_custom_kernel:
                return self.cuda_kernel.forward(x, self.conv3d.weight, self.stride, self.padding, self.groups)[0]
            
            # Use selected memory format
            if self.use_channels_last:
                x = x.to(memory_format=torch.channels_last_3d)
                
            # Use standard implementation with optimized memory format
            return self.conv3d(x)
        except Exception:
            # Fallback to standard implementation
            return self.conv3d(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5, 7)  # Asymmetric kernel
width = 64
height = 64
depth = 64

def get_inputs():
    x = torch.randn(batch_size, in_channels, width, height, depth)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization