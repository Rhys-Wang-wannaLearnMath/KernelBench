import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized model that performs a transposed convolution, adds a value,
    takes the minimum, applies GELU, and multiplies by a value.
    
    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        kernel_size (int): Size of the convolutional kernel
        stride (int): Stride of the convolution
        add_value (float): Value to add after convolution
        multiply_value (float): Value to multiply after GELU
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, add_value, multiply_value):
        super(ModelNew, self).__init__()
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
        self.add_value = add_value
        self.multiply_value = multiply_value
        
        # Initialize CUDA kernel for fused operations
        self.kernel = None
        if torch.cuda.is_available():
            self._init_cuda_kernel()
            
            # Pre-convert weights to channels_last format for better performance
            self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(memory_format=torch.channels_last)
    
    def _init_cuda_kernel(self):
        """Initialize the CUDA kernel for fused operations"""
        cuda_source = """
        extern "C" __global__ void fused_post_processing(
            float* __restrict__ output,
            const float* __restrict__ input,
            const int numel,
            const float add_value,
            const float multiply_value)
        {
            // Calculate global thread position
            const int idx = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Constants for GELU approximation
            const float sqrt_2_over_pi = 0.7978845608028654f;
            const float coef = 0.044715f;
            
            // Each thread processes multiple elements for better efficiency
            const int stride = blockDim.x * gridDim.x;
            
            // Process 4 elements at a time when possible (vector loading)
            int i = idx * 4;
            for (; i + 3 < numel; i += stride * 4) {
                // Check if memory is aligned for vectorized access
                if (((uintptr_t)&input[i] & 15) == 0 && ((uintptr_t)&output[i] & 15) == 0) {
                    // Load 4 input values at once with aligned access
                    float4 val4 = *((float4*)&input[i]);
                    
                    // Process each element
                    float val1 = val4.x + add_value;
                    float val2 = val4.y + add_value;
                    float val3 = val4.z + add_value;
                    float val4_w = val4.w + add_value;
                    
                    // Min operation
                    val1 = fminf(val1, 0.0f);
                    val2 = fminf(val2, 0.0f);
                    val3 = fminf(val3, 0.0f);
                    val4_w = fminf(val4_w, 0.0f);
                    
                    // GELU approximation
                    float val1_cubed = val1 * val1 * val1;
                    float val2_cubed = val2 * val2 * val2;
                    float val3_cubed = val3 * val3 * val3;
                    float val4_cubed = val4_w * val4_w * val4_w;
                    
                    float inner1 = sqrt_2_over_pi * (val1 + coef * val1_cubed);
                    float inner2 = sqrt_2_over_pi * (val2 + coef * val2_cubed);
                    float inner3 = sqrt_2_over_pi * (val3 + coef * val3_cubed);
                    float inner4 = sqrt_2_over_pi * (val4_w + coef * val4_cubed);
                    
                    float tanh_inner1 = tanhf(inner1);
                    float tanh_inner2 = tanhf(inner2);
                    float tanh_inner3 = tanhf(inner3);
                    float tanh_inner4 = tanhf(inner4);
                    
                    val1 = 0.5f * val1 * (1.0f + tanh_inner1);
                    val2 = 0.5f * val2 * (1.0f + tanh_inner2);
                    val3 = 0.5f * val3 * (1.0f + tanh_inner3);
                    val4_w = 0.5f * val4_w * (1.0f + tanh_inner4);
                    
                    // Multiply operation
                    val1 *= multiply_value;
                    val2 *= multiply_value;
                    val3 *= multiply_value;
                    val4_w *= multiply_value;
                    
                    // Store results using vectorized write
                    float4 out_val4;
                    out_val4.x = val1;
                    out_val4.y = val2;
                    out_val4.z = val3;
                    out_val4.w = val4_w;
                    *((float4*)&output[i]) = out_val4;
                } else {
                    // Unaligned access - process individually
                    for (int j = 0; j < 4 && i + j < numel; j++) {
                        float val = input[i + j];
                        
                        // Add operation
                        val += add_value;
                        
                        // Min operation
                        val = fminf(val, 0.0f);
                        
                        // GELU approximation
                        float val_cubed = val * val * val;
                        float inner = sqrt_2_over_pi * (val + coef * val_cubed);
                        float tanh_inner = tanhf(inner);
                        val = 0.5f * val * (1.0f + tanh_inner);
                        
                        // Multiply operation
                        val *= multiply_value;
                        
                        // Store result
                        output[i + j] = val;
                    }
                }
            }
            
            // Process remaining elements
            for (; i < numel; i += stride) {
                if (i < numel) {
                    float val = input[i];
                    
                    // Add operation
                    val += add_value;
                    
                    // Min operation
                    val = fminf(val, 0.0f);
                    
                    // GELU approximation
                    float val_cubed = val * val * val;
                    float inner = sqrt_2_over_pi * (val + coef * val_cubed);
                    float tanh_inner = tanhf(inner);
                    val = 0.5f * val * (1.0f + tanh_inner);
                    
                    // Multiply operation
                    val *= multiply_value;
                    
                    // Store result
                    output[i] = val;
                }
            }
        }
        """
        
        from torch.utils.cpp_extension import load_inline
        
        try:
            self.kernel = load_inline(
                name="fused_ops",
                cpp_sources="""
                #include <torch/extension.h>
                
                // Forward declarations
                extern "C" __global__ void fused_post_processing(
                    float* output,
                    const float* input,
                    const int numel,
                    const float add_value,
                    const float multiply_value);
                
                torch::Tensor fused_ops(torch::Tensor input, float add_value, float multiply_value) {
                    // Check input tensor
                    TORCH_CHECK(input.is_cuda(), "Input tensor must be on CUDA device");
                    TORCH_CHECK(input.is_contiguous(), "Input tensor must be contiguous");
                    TORCH_CHECK(input.dtype() == torch::kFloat32, "Input tensor must be float32");
                    
                    // Get tensor dimensions
                    int numel = input.numel();
                    
                    // Create output tensor
                    auto output = torch::empty_like(input);
                    
                    // Configure kernel launch parameters
                    const int threads = 256;
                    const int blocks = std::min(65535, (numel / 4 + threads - 1) / threads);
                    
                    // Launch kernel
                    fused_post_processing<<<blocks, threads>>>(
                        output.data_ptr<float>(),
                        input.data_ptr<float>(),
                        numel,
                        add_value,
                        multiply_value
                    );
                    
                    return output;
                }
                
                PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
                    m.def("fused_ops", &fused_ops, "Fused post-processing operations");
                }
                """,
                cuda_sources=cuda_source,
                extra_cuda_cflags=['-O3', '--use_fast_math'],
                functions=["fused_ops"],
                verbose=False
            )
        except Exception as e:
            print(f"Failed to load CUDA kernel: {e}")
            self.kernel = None

    def forward(self, x):
        # Use channels_last memory format for better performance on modern GPUs
        if x.is_cuda and x.dim() == 4:
            try:
                # Enable cuDNN benchmarking to find the best algorithm
                with torch.backends.cudnn.flags(enabled=True, benchmark=True, deterministic=False):
                    # Convert to channels_last format
                    x_cl = x.to(memory_format=torch.channels_last)
                    
                    # Perform the convolution transpose operation
                    output = torch.nn.functional.conv_transpose2d(
                        x_cl, 
                        self.conv_transpose.weight,
                        self.conv_transpose.bias,
                        self.conv_transpose.stride,
                        self.conv_transpose.padding,
                        self.conv_transpose.output_padding,
                        self.conv_transpose.groups,
                        self.conv_transpose.dilation
                    )
                
                # Apply fused post-processing operations using our CUDA kernel
                if self.kernel is not None:
                    try:
                        # Ensure output is contiguous for the CUDA kernel
                        output_contiguous = output.contiguous()
                        return self.kernel.fused_ops(output_contiguous, self.add_value, self.multiply_value)
                    except Exception:
                        # Fallback to standard PyTorch operations if kernel fails
                        pass
                
                # Standard PyTorch operations if CUDA kernel isn't available or fails
                output = output + self.add_value
                output = torch.min(output, torch.tensor(0.0, device=output.device))
                output = torch.nn.functional.gelu(output)
                output = output * self.multiply_value
                return output
                
            except Exception:
                # Fallback to standard implementation if channels_last optimization fails
                pass
        
        # Standard implementation using PyTorch's built-in operations
        x = self.conv_transpose(x)
        x = x + self.add_value
        x = torch.min(x, torch.tensor(0.0, device=x.device))
        x = torch.nn.functional.gelu(x)
        x = x * self.multiply_value
        return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 32
out_channels = 16
height, width = 32, 32
kernel_size = 4
stride = 2
add_value = 0.5
multiply_value = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_channels, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, add_value, multiply_value]