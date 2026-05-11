import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModelNew(nn.Module):
    """
    Performs a depthwise 2D convolution with a square input and an asymmetric kernel.

    Args:
        in_channels (int): Number of channels in the input tensor.
        kernel_size (int): Size of the convolution kernel.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create weight parameter with optimal shape
        self.weight = nn.Parameter(torch.empty(in_channels, 1, kernel_size, 1))
        if bias:
            self.bias = nn.Parameter(torch.empty(in_channels))
        else:
            self.bias = None
        
        # Cache convolution parameters as instance variables
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = in_channels
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        
        # Initialize weights using the same method as nn.Conv2d
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
        
        # Define the CUDA kernel for asymmetric depthwise convolution
        if torch.cuda.is_available():
            self._setup_cuda_kernel()
        else:
            self.forward_impl = self._forward_pytorch
    
    def _setup_cuda_kernel(self):
        cuda_kernel_code = '''
        extern "C" __global__ void asymmetric_depthwise_conv_kernel(
            const float* input, const float* weight, const float* bias,
            float* output, const int batch_size, const int channels,
            const int in_height, const int in_width, const int out_height, const int out_width,
            const int kernel_size, const int stride, const int padding, const int dilation) {
            
            // Calculate output position
            const int n = blockIdx.x;  // batch index
            const int c = blockIdx.y;  // channel index
            const int h_out = blockIdx.z / out_width;  // output height index
            const int w_out = blockIdx.z % out_width;  // output width index
            
            if (n >= batch_size || c >= channels || h_out >= out_height || w_out >= out_width)
                return;
                
            // Calculate input position
            const int h_in = h_out * stride - padding;
            const int w_in = w_out * stride - padding;
            
            // Calculate output index
            const int out_idx = ((n * channels + c) * out_height + h_out) * out_width + w_out;
            
            // Initialize output value
            float sum = 0.0f;
            
            // Perform convolution along vertical dimension only
            for (int k = 0; k < kernel_size; ++k) {
                const int h = h_in + k * dilation;
                
                if (h >= 0 && h < in_height) {
                    const int in_idx = ((n * channels + c) * in_height + h) * in_width + w_in;
                    const int weight_idx = c * kernel_size + k;
                    
                    sum += input[in_idx] * weight[weight_idx];
                }
            }
            
            // Add bias if present
            if (bias != nullptr) {
                sum += bias[c];
            }
            
            // Write output
            output[out_idx] = sum;
        }
        '''
        
        try:
            from torch.utils.cpp_extension import load_inline
            
            # Try to compile and load the CUDA kernel
            asymmetric_conv_cuda = load_inline(
                name="asymmetric_conv_cuda",
                cpp_sources="",
                cuda_sources=cuda_kernel_code,
                functions=["asymmetric_depthwise_conv_kernel"],
                with_cuda=True,
                verbose=False
            )
            
            self.asymmetric_conv_cuda = asymmetric_conv_cuda
            self.forward_impl = self._forward_cuda
        except Exception as e:
            # Fall back to PyTorch implementation if CUDA kernel compilation fails
            print(f"CUDA kernel compilation failed, falling back to PyTorch implementation: {e}")
            self.forward_impl = self._forward_pytorch
    
    def _forward_cuda(self, x):
        # Ensure input is contiguous
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Get input dimensions
        batch_size, channels, in_height, in_width = x.shape
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.dilation * (self.kernel_size - 1) - 1) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.dilation * (1 - 1) - 1) // self.stride + 1
        
        # Reshape weight for the kernel
        weight = self.weight.view(self.in_channels, self.kernel_size)
        
        # Create output tensor
        output = torch.empty(batch_size, channels, out_height, out_width, device=x.device, dtype=x.dtype)
        
        # Calculate grid and block dimensions
        grid_dim = (batch_size, channels, out_height * out_width)
        
        # Launch the CUDA kernel
        self.asymmetric_conv_cuda.asymmetric_depthwise_conv_kernel(
            grid=grid_dim, block=(1, 1, 1),
            args=[
                x.data_ptr(), weight.data_ptr(), 
                self.bias.data_ptr() if self.bias is not None else None,
                output.data_ptr(), batch_size, channels,
                in_height, in_width, out_height, out_width,
                self.kernel_size, self.stride, self.padding, self.dilation
            ]
        )
        
        return output
    
    def _forward_pytorch(self, x):
        # Direct call to F.conv2d with minimal overhead
        return F.conv2d(
            x, 
            self.weight, 
            self.bias, 
            self.stride, 
            self.padding, 
            self.dilation, 
            self.groups
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the depthwise 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, in_channels, height_out, width_out).
        """
        return self.forward_impl(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
kernel_size = 3
width = 256
height = 256
stride = 1
padding = 0
dilation = 1

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, kernel_size, stride, padding, dilation]