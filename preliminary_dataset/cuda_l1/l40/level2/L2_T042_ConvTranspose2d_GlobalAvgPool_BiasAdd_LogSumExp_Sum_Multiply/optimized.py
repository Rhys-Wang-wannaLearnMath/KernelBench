import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, global average pooling, 
    adds a bias, applies log-sum-exp, sum, and multiplication.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias_shape):
        super(ModelNew, self).__init__()
        # Create a standard ConvTranspose2d layer to initialize weights properly
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        # Extract the weight
        self.weight = nn.Parameter(self.conv_transpose.weight.data)
        # Initialize bias separately to match the reference implementation
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Remove the original conv_transpose to avoid confusion
        delattr(self, 'conv_transpose')
        
        # Flag to track if we're using the CUDA kernel
        self.use_cuda_kernel = False
        
        # Try to load the custom CUDA extension
        if torch.cuda.is_available():
            try:
                # Write the source code to files
                with open("fused_conv_extension.cpp", "w") as f:
                    f.write(self._get_cpp_source())
                with open("fused_conv_kernel.cu", "w") as f:
                    f.write(self._get_cuda_source())
                
                # Load the extension
                self.fused_conv = load(
                    name="fused_conv",
                    sources=["fused_conv_extension.cpp", "fused_conv_kernel.cu"],
                    verbose=False
                )
                self.use_cuda_kernel = True
            except Exception as e:
                print(f"Failed to load CUDA extension: {e}")
                self.use_cuda_kernel = False
    
    def _get_cpp_source(self):
        return """
        #include <torch/extension.h>
        #include <vector>

        // Forward declarations of CUDA functions
        torch::Tensor conv_transpose_fused_cuda(
            const torch::Tensor& input,
            const torch::Tensor& weight,
            const torch::Tensor& bias);

        // C++ interface
        torch::Tensor conv_transpose_fused(
            const torch::Tensor& input,
            const torch::Tensor& weight,
            const torch::Tensor& bias) {
            
            return conv_transpose_fused_cuda(input, weight, bias);
        }

        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
            m.def("conv_transpose_fused", &conv_transpose_fused, "Fused ConvTranspose2d operations");
        }
        """
    
    def _get_cuda_source(self):
        return """
        #include <torch/extension.h>
        #include <cuda.h>
        #include <cuda_runtime.h>
        #include <vector>
        #include <ATen/cuda/CUDAContext.h>

        // Kernel for transposed convolution
        template <typename scalar_t>
        __global__ void conv_transpose_kernel(
            const scalar_t* __restrict__ input,
            const scalar_t* __restrict__ weight,
            scalar_t* __restrict__ output,
            const int batch_size,
            const int in_channels,
            const int out_channels,
            const int input_height,
            const int input_width,
            const int kernel_size,
            const int output_height,
            const int output_width) {
            
            const int b = blockIdx.z;
            const int oc = blockIdx.y;
            const int oh = blockIdx.x / output_width;
            const int ow = blockIdx.x % output_width;
            
            if (oh >= output_height || ow >= output_width) return;
            
            scalar_t value = 0;
            
            // Calculate input position range that affects this output position
            const int ih_start = max(0, oh - (kernel_size - 1));
            const int ih_end = min(input_height, oh + 1);
            const int iw_start = max(0, ow - (kernel_size - 1));
            const int iw_end = min(input_width, ow + 1);
            
            for (int ic = 0; ic < in_channels; ++ic) {
                for (int ih = ih_start; ih < ih_end; ++ih) {
                    for (int iw = iw_start; iw < iw_end; ++iw) {
                        // Calculate kernel position
                        const int kh = oh - ih;
                        const int kw = ow - iw;
                        
                        // For transposed convolution, we need to flip the kernel indices
                        const int input_idx = b * in_channels * input_height * input_width +
                                            ic * input_height * input_width +
                                            ih * input_width + iw;
                        
                        const int weight_idx = ic * out_channels * kernel_size * kernel_size +
                                             oc * kernel_size * kernel_size +
                                             (kernel_size - 1 - kh) * kernel_size + (kernel_size - 1 - kw);
                        
                        value += input[input_idx] * weight[weight_idx];
                    }
                }
            }
            
            const int output_idx = b * out_channels * output_height * output_width +
                                 oc * output_height * output_width +
                                 oh * output_width + ow;
            
            output[output_idx] = value;
        }

        // Kernel for average pooling, bias addition, logsumexp, sum, and multiplication
        template <typename scalar_t>
        __global__ void post_processing_kernel(
            const scalar_t* __restrict__ conv_output,
            const scalar_t* __restrict__ bias,
            scalar_t* __restrict__ final_output,
            const int batch_size,
            const int out_channels,
            const int output_height,
            const int output_width) {
            
            const int b = blockIdx.x * blockDim.x + threadIdx.x;
            
            if (b >= batch_size) return;
            
            // Use shared memory for intermediate results
            extern __shared__ scalar_t shared_mem[];
            scalar_t* channel_avgs = &shared_mem[threadIdx.x * out_channels];
            
            // First, compute average pooling and add bias for each channel
            for (int oc = 0; oc < out_channels; ++oc) {
                scalar_t sum = 0;
                const int pixels = output_height * output_width;
                
                // Calculate average for each channel
                for (int oh = 0; oh < output_height; ++oh) {
                    for (int ow = 0; ow < output_width; ++ow) {
                        const int idx = b * out_channels * pixels +
                                      oc * pixels +
                                      oh * output_width + ow;
                        sum += conv_output[idx];
                    }
                }
                
                // Average pooling
                channel_avgs[oc] = sum / pixels;
                
                // Add bias
                channel_avgs[oc] += bias[oc];
            }
            
            // Find max for numerical stability in logsumexp
            scalar_t max_val = channel_avgs[0];
            for (int oc = 1; oc < out_channels; ++oc) {
                max_val = max(max_val, channel_avgs[oc]);
            }
            
            // Compute logsumexp
            scalar_t sum_exp = 0;
            for (int oc = 0; oc < out_channels; ++oc) {
                sum_exp += exp(channel_avgs[oc] - max_val);
            }
            
            // Final result: log(sum(exp)) + max, then multiply by 10.0
            final_output[b] = (log(sum_exp) + max_val) * 10.0;
        }

        torch::Tensor conv_transpose_fused_cuda(
            const torch::Tensor& input,
            const torch::Tensor& weight,
            const torch::Tensor& bias) {
            
            const auto batch_size = input.size(0);
            const auto in_channels = input.size(1);
            const auto input_height = input.size(2);
            const auto input_width = input.size(3);
            
            const auto out_channels = weight.size(1);
            const auto kernel_size = weight.size(2);
            
            // Calculate output dimensions for transposed convolution
            const auto output_height = input_height + kernel_size - 1;
            const auto output_width = input_width + kernel_size - 1;
            
            // Allocate memory for convolution output
            auto conv_output = torch::zeros({batch_size, out_channels, output_height, output_width},
                                          input.options());
            
            // Allocate memory for final output
            auto final_output = torch::zeros({batch_size, 1},
                                           input.options());
            
            // Set up grid and blocks for convolution kernel
            const dim3 blocks_conv(output_height * output_width, out_channels, batch_size);
            const dim3 threads_conv(1, 1, 1);
            
            // Set up grid and blocks for post-processing kernel
            const int threads_per_block = 128;
            const dim3 blocks_post((batch_size + threads_per_block - 1) / threads_per_block);
            const dim3 threads_post(threads_per_block);
            
            // Calculate shared memory size for post-processing
            const int shared_mem_size = threads_per_block * out_channels * sizeof(float);
            
            // Get CUDA stream
            cudaStream_t stream = at::cuda::getCurrentCUDAStream();
            
            // Launch kernels
            AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "conv_transpose_fused_cuda", ([&] {
                conv_transpose_kernel<scalar_t><<<blocks_conv, threads_conv, 0, stream>>>(
                    input.data_ptr<scalar_t>(),
                    weight.data_ptr<scalar_t>(),
                    conv_output.data_ptr<scalar_t>(),
                    batch_size,
                    in_channels,
                    out_channels,
                    input_height,
                    input_width,
                    kernel_size,
                    output_height,
                    output_width);
                
                post_processing_kernel<scalar_t><<<blocks_post, threads_post, shared_mem_size, stream>>>(
                    conv_output.data_ptr<scalar_t>(),
                    bias.data_ptr<scalar_t>(),
                    final_output.data_ptr<scalar_t>(),
                    batch_size,
                    out_channels,
                    output_height,
                    output_width);
            }));
            
            return final_output;
        }
        """
    
    def forward(self, x):
        # Use our custom fused operation if available and on CUDA
        if self.use_cuda_kernel and x.is_cuda:
            # Reshape bias to match the kernel's expectation
            bias_reshaped = self.bias.view(-1)
            return self.fused_conv.conv_transpose_fused(x, self.weight, bias_reshaped)
        else:
            # Fallback to standard PyTorch operations
            x = F.conv_transpose2d(x, self.weight, bias=None, stride=1, padding=0)
            x = torch.mean(x, dim=(2, 3), keepdim=True)  # Global average pooling
            x = x + self.bias
            x = torch.logsumexp(x, dim=1, keepdim=True)  # Log-sum-exp
            x = torch.sum(x, dim=(2, 3))  # Sum
            x = x * 10.0  # Multiplication
            return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
height, width = 32, 32
kernel_size = 3
bias_shape = (out_channels, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, bias_shape]