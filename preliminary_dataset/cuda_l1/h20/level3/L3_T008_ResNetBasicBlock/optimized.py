import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define CUDA kernels for optimized operations
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel for fused residual addition and ReLU
template <typename scalar_t>
__global__ void fused_residual_add_relu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ output,
    int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Grid-stride loop for better occupancy
    for (int i = idx; i < size; i += stride) {
        const scalar_t sum = input[i] + residual[i];
        output[i] = sum > scalar_t(0) ? sum : scalar_t(0);
    }
}

// Optimized kernel using float4 vectorization when possible
__global__ void fused_residual_add_relu_float4_kernel(
    const float4* __restrict__ input,
    const float4* __restrict__ residual,
    float4* __restrict__ output,
    int vec_size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process 4 elements at a time
    for (int i = idx; i < vec_size; i += stride) {
        float4 in_val = input[i];
        float4 res_val = residual[i];
        float4 out_val;
        
        // Process 4 elements in parallel
        out_val.x = (in_val.x + res_val.x) > 0.0f ? (in_val.x + res_val.x) : 0.0f;
        out_val.y = (in_val.y + res_val.y) > 0.0f ? (in_val.y + res_val.y) : 0.0f;
        out_val.z = (in_val.z + res_val.z) > 0.0f ? (in_val.z + res_val.z) : 0.0f;
        out_val.w = (in_val.w + res_val.w) > 0.0f ? (in_val.w + res_val.w) : 0.0f;
        
        output[i] = out_val;
    }
}

// Specialized kernel for ResNet's 224x224 images with NCHW format
__global__ void fused_residual_add_relu_resnet_kernel(
    const float* __restrict__ input,
    const float* __restrict__ residual,
    float* __restrict__ output,
    int batch_size,
    int channels,
    int height,
    int width) {
    
    // Use shared memory to cache data
    __shared__ float in_tile[16][16];
    __shared__ float res_tile[16][16];
    
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int bz = blockIdx.z;
    
    // Calculate indices
    const int c = bz % channels;
    const int n = bz / channels;
    
    const int h_start = by * 16;
    const int w_start = bx * 16;
    
    // Check if we're within bounds
    if (n < batch_size && c < channels) {
        const int h = h_start + ty;
        const int w = w_start + tx;
        
        if (h < height && w < width) {
            const int idx = ((n * channels + c) * height + h) * width + w;
            
            // Load data into shared memory
            in_tile[ty][tx] = input[idx];
            res_tile[ty][tx] = residual[idx];
            
            // Ensure all threads have loaded their data
            __syncthreads();
            
            // Process and write results back to global memory
            const float sum = in_tile[ty][tx] + res_tile[ty][tx];
            output[idx] = sum > 0.0f ? sum : 0.0f;
        }
    }
}

torch::Tensor fused_residual_add_relu(torch::Tensor input, torch::Tensor residual) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(residual.device().is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(input.sizes() == residual.sizes(), "input and residual must have the same shape");
    
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Ensure tensors are contiguous
    auto input_contig = input.contiguous();
    auto residual_contig = residual.contiguous();
    
    // Use specialized kernel for 4D tensors in NCHW format with 224x224 dimensions (common in ResNet)
    if (input.dim() == 4 && input.scalar_type() == torch::kFloat && 
        input.size(2) == 224 && input.size(3) == 224) {
        const int batch_size = input.size(0);
        const int channels = input.size(1);
        const int height = input.size(2);
        const int width = input.size(3);
        
        dim3 threads(16, 16);
        dim3 blocks(
            (width + threads.x - 1) / threads.x,
            (height + threads.y - 1) / threads.y,
            batch_size * channels
        );
        
        fused_residual_add_relu_resnet_kernel<<<blocks, threads>>>(
            input_contig.data_ptr<float>(),
            residual_contig.data_ptr<float>(),
            output.data_ptr<float>(),
            batch_size,
            channels,
            height,
            width
        );
        return output;
    }
    
    // Use vectorized kernel for float tensors when size is divisible by 4
    if (input.scalar_type() == torch::kFloat && size % 4 == 0) {
        const int vec_size = size / 4;
        const int threads = 256;
        const int blocks = std::min(65535, (vec_size + threads - 1) / threads);
        
        fused_residual_add_relu_float4_kernel<<<blocks, threads>>>(
            reinterpret_cast<const float4*>(input_contig.data_ptr<float>()),
            reinterpret_cast<const float4*>(residual_contig.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            vec_size
        );
    } 
    // Fallback to generic kernel
    else {
        const int threads = 256;
        const int blocks = std::min(65535, (size + threads - 1) / threads);
        
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_residual_add_relu", ([&] {
            fused_residual_add_relu_kernel<scalar_t><<<blocks, threads>>>(
                input_contig.data_ptr<scalar_t>(),
                residual_contig.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                size
            );
        }));
    }
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_residual_add_relu", &fused_residual_add_relu, "Optimized Fused Residual Addition and ReLU");
}
"""

# Try to load the custom CUDA extension
try:
    resnet_cuda = load_inline(
        name="resnet_cuda_opt",
        cpp_sources="",
        cuda_sources=cuda_source,
        functions=["fused_residual_add_relu"],
        with_cuda=True,
        extra_cuda_cflags=["-O3", "--use_fast_math"]
    )
except Exception as e:
    print(f"Failed to load CUDA extension: {e}")
    resnet_cuda = None

class FusedConvBN(nn.Module):
    """
    Fused convolution and batch normalization for inference
    """
    def __init__(self, conv, bn):
        super(FusedConvBN, self).__init__()
        
        # Store original parameters
        self.conv_weight = conv.weight
        self.conv_bias = getattr(conv, 'bias', None)
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        
        # Store batch norm parameters
        self.bn_weight = bn.weight
        self.bn_bias = bn.bias
        self.bn_running_mean = bn.running_mean
        self.bn_running_var = bn.running_var
        self.bn_eps = bn.eps
        
        # Pre-compute fused parameters
        self._compute_fused_params()
        
    def _compute_fused_params(self):
        """Compute fused conv+bn parameters"""
        if self.conv_bias is None:
            self.conv_bias = torch.zeros_like(self.bn_running_mean)
            
        # Compute fused parameters
        inv_std = torch.rsqrt(self.bn_running_var + self.bn_eps)
        
        # Reshape for broadcasting
        bn_weight_view = self.bn_weight.reshape([-1] + [1] * (len(self.conv_weight.shape) - 1))
        inv_std_view = inv_std.reshape([-1] + [1] * (len(self.conv_weight.shape) - 1))
        
        # Fuse parameters
        self.register_buffer('fused_weight', self.conv_weight * (bn_weight_view * inv_std_view))
        self.register_buffer('fused_bias', (self.conv_bias - self.bn_running_mean) * inv_std * self.bn_weight + self.bn_bias)
        
    def forward(self, x):
        # Ensure tensors are contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
            
        return F.conv2d(x, self.fused_weight, self.fused_bias, 
                       self.stride, self.padding, self.dilation, self.groups)

class ModelNew(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        """
        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :param stride: Stride for the first convolutional layer
        """
        super(ModelNew, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion),
        )
        self.stride = stride
        
        # Create fused modules for inference
        self.fused_modules_initialized = False
        self.use_cuda_kernel = resnet_cuda is not None

    def _initialize_fused_modules(self):
        """Initialize fused modules on first forward pass"""
        if not self.fused_modules_initialized:
            self.fused_conv1_bn1 = FusedConvBN(self.conv1, self.bn1)
            self.fused_conv2_bn2 = FusedConvBN(self.conv2, self.bn2)
            self.fused_downsample = FusedConvBN(self.downsample[0], self.downsample[1])
            self.fused_modules_initialized = True

    def forward(self, x):
        """
        :param x: Input tensor, shape (batch_size, in_channels, height, width)
        :return: Output tensor, shape (batch_size, out_channels, height, width)
        """
        # Initialize fused modules if not already done
        if not self.fused_modules_initialized:
            self._initialize_fused_modules()
            
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Main path with fused operations
        out = self.fused_conv1_bn1(x)
        out = F.relu(out, inplace=True)
        out = self.fused_conv2_bn2(out)
        
        # Downsample path
        identity = self.fused_downsample(x)
        
        # Use optimized CUDA kernel for residual addition and ReLU if available
        if self.use_cuda_kernel and x.is_cuda:
            try:
                return resnet_cuda.fused_residual_add_relu(out, identity)
            except Exception as e:
                # Fallback to PyTorch operations
                out = out + identity
                return F.relu(out, inplace=True)
        else:
            # Standard PyTorch operations
            out = out + identity
            return F.relu(out, inplace=True)

# Test code
in_channels = 3
out_channels = 64
stride = 1
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, stride]