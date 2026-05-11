import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Optimized model that performs a 3D convolution, divides by a constant, applies max pooling,
    global average pooling, adds a bias term, and sums along a specific dimension.
    """
    def __init__(self, in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim):
        super(ModelNew, self).__init__()
        # Create convolution layer
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size)
        
        # Pre-scale weights and bias by 1/divisor to fuse the division operation
        with torch.no_grad():
            self.conv.weight.div_(divisor)
            if self.conv.bias is not None:
                self.conv.bias.div_(divisor)
        
        # Store other parameters
        self.divisor = divisor
        self.max_pool = nn.MaxPool3d(pool_size)
        self.global_avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.sum_dim = sum_dim
        
        # Enable cuDNN benchmarking for optimal algorithm selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            # Disable deterministic algorithms for better performance
            torch.backends.cudnn.deterministic = False
            # Convert weights to channels_last format for better memory access
            self.conv.weight.data = self.conv.weight.data.to(memory_format=torch.channels_last_3d)
            
            # Register CUDA kernels
            self._setup_cuda_kernels()
    
    def _setup_cuda_kernels(self):
        if not torch.cuda.is_available():
            return
            
        self.kernel_code = """
        extern "C" __global__ void fused_conv3d_maxpool_kernel(
            const float* __restrict__ input,
            const float* __restrict__ weight,
            const float* __restrict__ bias,
            float* __restrict__ output,
            int batch_size, int in_channels, int out_channels,
            int in_depth, int in_height, int in_width,
            int kernel_d, int kernel_h, int kernel_w,
            int out_depth, int out_height, int out_width,
            int pool_d, int pool_h, int pool_w,
            int pool_out_depth, int pool_out_height, int pool_out_width)
        {
            // Get output position
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= batch_size * out_channels * pool_out_depth * pool_out_height * pool_out_width)
                return;
                
            // Calculate position in pooled output
            const int pw = tid % pool_out_width;
            const int ph = (tid / pool_out_width) % pool_out_height;
            const int pd = (tid / (pool_out_width * pool_out_height)) % pool_out_depth;
            const int oc = (tid / (pool_out_width * pool_out_height * pool_out_depth)) % out_channels;
            const int b = tid / (pool_out_width * pool_out_height * pool_out_depth * out_channels);
            
            // Calculate corresponding region in conv output
            const int start_d = pd * pool_d;
            const int start_h = ph * pool_h;
            const int start_w = pw * pool_w;
            
            // Perform max pooling over the conv output region
            float max_val = -FLT_MAX;
            
            // For each position in the pooling window
            for (int d = 0; d < pool_d && (start_d + d) < out_depth; ++d) {
                for (int h = 0; h < pool_h && (start_h + h) < out_height; ++h) {
                    for (int w = 0; w < pool_w && (start_w + w) < out_width; ++w) {
                        // Calculate position in conv output
                        const int od = start_d + d;
                        const int oh = start_h + h;
                        const int ow = start_w + w;
                        
                        // Compute convolution for this position
                        float conv_result = 0.0f;
                        if (bias != nullptr) {
                            conv_result = bias[oc];
                        }
                        
                        // For each input channel
                        for (int ic = 0; ic < in_channels; ++ic) {
                            // For each position in the kernel
                            for (int kd = 0; kd < kernel_d; ++kd) {
                                const int id = od + kd - kernel_d / 2;
                                if (id < 0 || id >= in_depth) continue;
                                
                                for (int kh = 0; kh < kernel_h; ++kh) {
                                    const int ih = oh + kh - kernel_h / 2;
                                    if (ih < 0 || ih >= in_height) continue;
                                    
                                    for (int kw = 0; kw < kernel_w; ++kw) {
                                        const int iw = ow + kw - kernel_w / 2;
                                        if (iw < 0 || iw >= in_width) continue;
                                        
                                        // Get input value
                                        const int input_idx = b * (in_channels * in_depth * in_height * in_width) +
                                                             ic * (in_depth * in_height * in_width) +
                                                             id * (in_height * in_width) +
                                                             ih * in_width +
                                                             iw;
                                        const float input_val = input[input_idx];
                                        
                                        // Get weight value
                                        const int weight_idx = oc * (in_channels * kernel_d * kernel_h * kernel_w) +
                                                              ic * (kernel_d * kernel_h * kernel_w) +
                                                              kd * (kernel_h * kernel_w) +
                                                              kh * kernel_w +
                                                              kw;
                                        const float weight_val = weight[weight_idx];
                                        
                                        // Accumulate
                                        conv_result += input_val * weight_val;
                                    }
                                }
                            }
                        }
                        
                        // Update max value
                        max_val = fmaxf(max_val, conv_result);
                    }
                }
            }
            
            // Write output
            output[tid] = max_val;
        }
        """
        
        # Note: In a real implementation, we would compile this kernel using torch.utils.cpp_extension
        # or similar. For this exercise, we'll use PyTorch's built-in operations instead.

    def forward(self, x):
        # Convert to channels_last memory format for better performance on CUDA
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last_3d)
        
        # Apply convolution (with pre-scaled weights, division is fused)
        x = self.conv(x)
        
        # Apply max pooling
        x = self.max_pool(x)
        
        # Apply global average pooling
        x = self.global_avg_pool(x)
        
        # Add bias (after pooling to operate on smaller tensor)
        x = x + self.bias
        
        # Sum along specified dimension
        x = torch.sum(x, dim=self.sum_dim)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 16, 32, 32
kernel_size = (3, 3, 3)
divisor = 2.0
pool_size = (2, 2, 2)
bias_shape = (out_channels, 1, 1, 1)
sum_dim = 1

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [in_channels, out_channels, kernel_size, divisor, pool_size, bias_shape, sum_dim]