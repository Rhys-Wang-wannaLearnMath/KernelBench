import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    An optimized model that performs a 3D transposed convolution, followed by batch normalization,
    two average pooling layers.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias_shape):
        super(ModelNew, self).__init__()
        
        # Create reference modules to ensure identical initialization
        self.conv_transpose = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(out_channels)
        self.avg_pool1 = nn.AvgPool3d(kernel_size=2)
        self.avg_pool2 = nn.AvgPool3d(kernel_size=2)
        
        # Enable cudnn benchmark for kernel autotuning
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # Check if torch.compile is available (PyTorch 2.0+)
        self.use_compile = hasattr(torch, 'compile')
        if self.use_compile:
            try:
                self.optimized_forward = torch.compile(self._optimized_forward)
            except:
                self.use_compile = False
        
        # Cache for extracted parameters
        self._extracted_params = None
        
        # Try to create a JIT traced version of the forward pass
        self.use_jit = False
        try:
            if torch.cuda.is_available():
                dummy_input = torch.randn(1, in_channels, 8, 8, 8, device='cuda')
                self.traced_forward = torch.jit.trace(self._forward_for_jit, dummy_input)
                self.use_jit = True
        except:
            self.use_jit = False
    
    def _extract_parameters(self):
        """Extract parameters once to avoid repeated attribute access"""
        if self._extracted_params is None:
            self._extracted_params = {
                'weight': self.conv_transpose.weight,
                'bias': self.conv_transpose.bias,
                'bn_weight': self.batch_norm.weight,
                'bn_bias': self.batch_norm.bias,
                'running_mean': self.batch_norm.running_mean,
                'running_var': self.batch_norm.running_var,
                'eps': self.batch_norm.eps,
                'stride': self.conv_transpose.stride,
                'padding': self.conv_transpose.padding,
                'output_padding': self.conv_transpose.output_padding,
                'groups': self.conv_transpose.groups,
                'dilation': self.conv_transpose.dilation
            }
        return self._extracted_params
    
    def _forward_for_jit(self, x):
        """Optimized forward implementation for JIT tracing"""
        params = self._extract_parameters()
        
        # Step 1: ConvTranspose3d
        x = F.conv_transpose3d(
            x, params['weight'], params['bias'], 
            stride=params['stride'], 
            padding=params['padding'],
            output_padding=params['output_padding'],
            groups=params['groups'],
            dilation=params['dilation']
        )
        
        # Step 2: BatchNorm3d
        x = F.batch_norm(
            x,
            params['running_mean'],
            params['running_var'],
            params['bn_weight'],
            params['bn_bias'],
            False,  # Not training
            0.1,    # Default momentum
            params['eps']
        )
        
        # Step 3: Fused pooling - replace two consecutive AvgPool3d(kernel_size=2) 
        # with a single AvgPool3d(kernel_size=4, stride=4)
        x = F.avg_pool3d(x, kernel_size=4, stride=4)
        
        return x
    
    def _optimized_forward(self, x, weight, bias, bn_weight, bn_bias, running_mean, running_var, eps,
                          stride, padding, output_padding, groups, dilation):
        """Optimized forward implementation that can be compiled with torch.compile"""
        # Step 1: ConvTranspose3d
        x = F.conv_transpose3d(
            x, weight, bias, 
            stride=stride, 
            padding=padding,
            output_padding=output_padding,
            groups=groups,
            dilation=dilation
        )
        
        # Step 2: BatchNorm3d
        x = F.batch_norm(
            x,
            running_mean,
            running_var,
            bn_weight,
            bn_bias,
            False,  # Not training
            0.1,    # Default momentum
            eps
        )
        
        # Step 3: Fused pooling - replace two consecutive AvgPool3d(kernel_size=2) 
        # with a single AvgPool3d(kernel_size=4, stride=4)
        x = F.avg_pool3d(x, kernel_size=4, stride=4)
        
        return x
    
    def forward(self, x):
        # Ensure input is contiguous for better memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Try using channels_last_3d memory format if available
        try:
            if hasattr(torch.memory_format, 'channels_last_3d'):
                x = x.to(memory_format=torch.memory_format.channels_last_3d)
                self.conv_transpose.weight.data = self.conv_transpose.weight.data.to(memory_format=torch.memory_format.channels_last_3d)
        except Exception:
            pass
        
        # Extract parameters once to avoid repeated attribute access
        params = self._extract_parameters()
        
        with torch.no_grad():
            # Try JIT traced version first if available
            if self.use_jit and x.is_cuda:
                try:
                    return self.traced_forward(x)
                except Exception:
                    pass
            
            # Try torch.compile if available
            if self.use_compile:
                try:
                    return self.optimized_forward(
                        x, params['weight'], params['bias'], 
                        params['bn_weight'], params['bn_bias'], 
                        params['running_mean'], params['running_var'], 
                        params['eps'], params['stride'], params['padding'], 
                        params['output_padding'], params['groups'], params['dilation']
                    )
                except Exception:
                    pass
            
            # Fallback to functional API implementation
            # Step 1: ConvTranspose3d
            x = F.conv_transpose3d(
                x, params['weight'], params['bias'], 
                stride=params['stride'], 
                padding=params['padding'],
                output_padding=params['output_padding'],
                groups=params['groups'],
                dilation=params['dilation']
            )
            
            # Step 2: BatchNorm3d
            x = F.batch_norm(
                x,
                params['running_mean'],
                params['running_var'],
                params['bn_weight'],
                params['bn_bias'],
                False,  # Not training
                0.1,    # Default momentum
                params['eps']
            )
            
            # Step 3: Fused pooling - replace two consecutive AvgPool3d(kernel_size=2) 
            # with a single AvgPool3d(kernel_size=4, stride=4)
            x = F.avg_pool3d(x, kernel_size=4, stride=4)
            
            return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_channels = 3
out_channels = 16
depth, height, width = 32, 32, 32
kernel_size = 3
stride = 2
padding = 1
bias_shape = (out_channels, 1, 1, 1)

def get_inputs():
    return [torch.randn(batch_size, in_channels, depth, height, width)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, bias_shape]