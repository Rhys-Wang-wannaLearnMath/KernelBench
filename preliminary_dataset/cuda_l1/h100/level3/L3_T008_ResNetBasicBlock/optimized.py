import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

# Define CUDA kernel for fused residual addition and ReLU
cuda_source = """
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// Optimized kernel using float4 vectorization for better memory throughput
template <typename scalar_t>
__global__ void fused_residual_add_relu_kernel(
    const scalar_t* __restrict__ input,
    const scalar_t* __restrict__ residual,
    scalar_t* __restrict__ output,
    int size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process elements in grid-stride loop for better efficiency
    for (int i = idx; i < size; i += stride) {
        const scalar_t sum = input[i] + residual[i];
        output[i] = sum > 0 ? sum : 0;
    }
}

// Vectorized kernel using float4 for higher memory throughput
__global__ void fused_residual_add_relu_float4_kernel(
    const float4* __restrict__ input,
    const float4* __restrict__ residual,
    float4* __restrict__ output,
    int vec_size) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    
    // Process elements in grid-stride loop with vectorized operations
    for (int i = idx; i < vec_size; i += stride) {
        float4 in_val = input[i];
        float4 res_val = residual[i];
        float4 out_val;
        
        // Process 4 elements at once
        out_val.x = fmaxf(in_val.x + res_val.x, 0.0f);
        out_val.y = fmaxf(in_val.y + res_val.y, 0.0f);
        out_val.z = fmaxf(in_val.z + res_val.z, 0.0f);
        out_val.w = fmaxf(in_val.w + res_val.w, 0.0f);
        
        output[i] = out_val;
    }
}

torch::Tensor fused_residual_add_relu(torch::Tensor input, torch::Tensor residual) {
    TORCH_CHECK(input.device().is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(residual.device().is_cuda(), "residual must be a CUDA tensor");
    TORCH_CHECK(input.sizes() == residual.sizes(), "input and residual shapes must match");
    
    auto output = torch::empty_like(input);
    const int size = input.numel();
    
    // Optimize thread and block configuration
    const int threads = 256;
    const int blocks = std::min(65535, (size + threads - 1) / threads);
    
    // Use vectorized kernel for float tensors when conditions are met
    if (input.scalar_type() == torch::kFloat && 
        size >= 1024 && 
        size % 4 == 0 && 
        input.is_contiguous() && 
        residual.is_contiguous() &&
        reinterpret_cast<uintptr_t>(input.data_ptr<float>()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(residual.data_ptr<float>()) % 16 == 0 &&
        reinterpret_cast<uintptr_t>(output.data_ptr<float>()) % 16 == 0) {
        
        const int vec_size = size / 4;
        const int vec_blocks = std::min(65535, (vec_size + threads - 1) / threads);
        
        fused_residual_add_relu_float4_kernel<<<vec_blocks, threads>>>(
            reinterpret_cast<const float4*>(input.data_ptr<float>()),
            reinterpret_cast<const float4*>(residual.data_ptr<float>()),
            reinterpret_cast<float4*>(output.data_ptr<float>()),
            vec_size);
    } else {
        AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "fused_residual_add_relu", ([&] {
            fused_residual_add_relu_kernel<scalar_t><<<blocks, threads>>>(
                input.data_ptr<scalar_t>(),
                residual.data_ptr<scalar_t>(),
                output.data_ptr<scalar_t>(),
                size);
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
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"Failed to load CUDA extension: {e}")
    resnet_cuda = None
    CUDA_AVAILABLE = False

class FusedConvBN(nn.Module):
    """
    Fused convolution and batch normalization for improved performance
    """
    def __init__(self, conv, bn):
        super(FusedConvBN, self).__init__()
        
        # Store original convolution parameters
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation
        self.groups = conv.groups
        
        # Pre-compute fused parameters
        self._fuse_conv_bn_params(conv, bn)
        
    def _fuse_conv_bn_params(self, conv, bn):
        """Pre-compute fused convolution and batch normalization parameters"""
        # Get original parameters
        conv_w = conv.weight
        conv_b = torch.zeros(conv_w.size(0), device=conv_w.device) if conv.bias is None else conv.bias
        
        bn_rm = bn.running_mean
        bn_rv = bn.running_var
        bn_w = bn.weight
        bn_b = bn.bias
        bn_eps = bn.eps
        
        # Compute batch norm scaling factor
        bn_scale = bn_w * torch.rsqrt(bn_rv + bn_eps)
        
        # Compute fused weights and bias
        # Fused weight = conv_weight * bn_scale.view(-1, 1, 1, 1)
        fused_w = conv_w * bn_scale.view(-1, 1, 1, 1)
        
        # Fused bias = (conv_bias - bn_mean) * bn_scale + bn_bias
        fused_b = (conv_b - bn_rm) * bn_scale + bn_b
        
        # Register parameters as buffers
        self.register_buffer('weight', fused_w)
        self.register_buffer('bias', fused_b)
        
    def forward(self, x):
        """Forward pass with fused convolution and batch normalization"""
        return F.conv2d(x, self.weight, self.bias, 
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
        
        # Keep original layers for compatibility
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
        
        # Optimization flags
        self.use_cuda_kernel = CUDA_AVAILABLE
        self.fused_modules_initialized = False
        
    def _initialize_fused_modules(self):
        """Initialize fused modules on first forward pass to ensure parameters are on the correct device"""
        if not self.fused_modules_initialized:
            # Create fused conv+bn modules
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
            
        # Main path with fused operations
        out = self.fused_conv1_bn1(x)
        out = F.relu(out, inplace=True)
        out = self.fused_conv2_bn2(out)
        
        # Downsample path
        identity = self.fused_downsample(x)
        
        # Use optimized CUDA kernel for residual addition and ReLU if available
        if self.use_cuda_kernel and x.is_cuda and resnet_cuda is not None:
            try:
                # Ensure tensors are contiguous for optimal memory access
                if not out.is_contiguous():
                    out = out.contiguous()
                if not identity.is_contiguous():
                    identity = identity.contiguous()
                    
                return resnet_cuda.fused_residual_add_relu(out, identity)
            except Exception:
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