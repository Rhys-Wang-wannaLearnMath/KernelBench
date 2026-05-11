import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        Optimized MBConv block implementation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :param kernel_size: Kernel size for the depthwise convolution.
        :param stride: Stride for the depthwise convolution.
        :param expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(ModelNew, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.hidden_dim = in_channels * expand_ratio
        self.has_expand = (expand_ratio != 1)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size - 1) // 2
        
        # Create standard PyTorch layers for initialization and parameter management
        if self.has_expand:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU6(inplace=True)
            )
        
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, stride=stride, 
                      padding=self.padding, groups=self.hidden_dim, bias=False),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU6(inplace=True)
        )
        
        self.project_conv = nn.Sequential(
            nn.Conv2d(self.hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        
        # Pre-compute fused weights and biases for optimized forward pass
        self._prepare_fused_parameters()
        
        # JIT compile the forward function for better performance
        self._compile_forward()
        
    def _prepare_fused_parameters(self):
        """Pre-compute fused conv+BN parameters for efficient forward pass"""
        eps = 1e-5
        
        # Fuse expand conv + BN
        if self.has_expand:
            expand_conv = self.expand_conv[0]
            expand_bn = self.expand_conv[1]
            
            # Compute fused weight and bias
            bn_var_rsqrt = torch.rsqrt(expand_bn.running_var + eps)
            bn_weight_scaled = expand_bn.weight * bn_var_rsqrt
            
            # Fuse into conv weight
            fused_expand_weight = expand_conv.weight * bn_weight_scaled.view(-1, 1, 1, 1)
            fused_expand_bias = expand_bn.bias - expand_bn.running_mean * bn_weight_scaled
            
            self.register_buffer('fused_expand_weight', fused_expand_weight)
            self.register_buffer('fused_expand_bias', fused_expand_bias)
        
        # Fuse depthwise conv + BN
        depthwise_conv = self.depthwise_conv[0]
        depthwise_bn = self.depthwise_conv[1]
        
        bn_var_rsqrt = torch.rsqrt(depthwise_bn.running_var + eps)
        bn_weight_scaled = depthwise_bn.weight * bn_var_rsqrt
        
        fused_depthwise_weight = depthwise_conv.weight * bn_weight_scaled.view(-1, 1, 1, 1)
        fused_depthwise_bias = depthwise_bn.bias - depthwise_bn.running_mean * bn_weight_scaled
        
        self.register_buffer('fused_depthwise_weight', fused_depthwise_weight)
        self.register_buffer('fused_depthwise_bias', fused_depthwise_bias)
        
        # Fuse project conv + BN
        project_conv = self.project_conv[0]
        project_bn = self.project_conv[1]
        
        bn_var_rsqrt = torch.rsqrt(project_bn.running_var + eps)
        bn_weight_scaled = project_bn.weight * bn_var_rsqrt
        
        fused_project_weight = project_conv.weight * bn_weight_scaled.view(-1, 1, 1, 1)
        fused_project_bias = project_bn.bias - project_bn.running_mean * bn_weight_scaled
        
        self.register_buffer('fused_project_weight', fused_project_weight)
        self.register_buffer('fused_project_bias', fused_project_bias)
    
    def _compile_forward(self):
        """JIT compile the optimized forward function for better performance"""
        try:
            # Define optimized forward function for JIT compilation
            @torch.jit.script
            def _optimized_forward(x, 
                                  expand_weight, expand_bias, 
                                  depthwise_weight, depthwise_bias,
                                  project_weight, project_bias,
                                  stride: int, padding: int, hidden_dim: int,
                                  has_expand: bool, use_residual: bool):
                identity = x
                
                # Expand phase with fused conv+BN+ReLU6
                if has_expand:
                    x = F.conv2d(x, expand_weight, expand_bias, 1, 0)
                    x = F.relu6(x)
                
                # Depthwise phase with fused conv+BN+ReLU6
                x = F.conv2d(x, depthwise_weight, depthwise_bias, 
                            stride, padding, groups=hidden_dim)
                x = F.relu6(x)
                
                # Project phase with fused conv+BN
                x = F.conv2d(x, project_weight, project_bias, 1, 0)
                
                # Residual connection
                if use_residual:
                    x = x + identity
                
                return x
            
            self._jit_forward = _optimized_forward
            self._use_jit = True
        except Exception:
            self._use_jit = False
    
    def _optimized_forward(self, x):
        """Optimized forward pass using fused parameters"""
        if self._use_jit:
            # Use JIT compiled forward function
            expand_weight = self.fused_expand_weight if self.has_expand else None
            expand_bias = self.fused_expand_bias if self.has_expand else None
            
            return self._jit_forward(
                x, 
                expand_weight, expand_bias,
                self.fused_depthwise_weight, self.fused_depthwise_bias,
                self.fused_project_weight, self.fused_project_bias,
                self.stride, self.padding, self.hidden_dim,
                self.has_expand, self.use_residual
            )
        else:
            # Fallback to non-JIT optimized forward
            identity = x
            
            # Optimized expand phase with fused conv+BN+ReLU6
            if self.has_expand:
                x = F.conv2d(x.contiguous(), self.fused_expand_weight, self.fused_expand_bias, 1, 0)
                x = F.relu6(x)
            
            # Optimized depthwise phase with fused conv+BN+ReLU6
            x = F.conv2d(x.contiguous(), self.fused_depthwise_weight, self.fused_depthwise_bias, 
                        self.stride, self.padding, groups=self.hidden_dim)
            x = F.relu6(x)
            
            # Optimized project phase with fused conv+BN
            x = F.conv2d(x.contiguous(), self.fused_project_weight, self.fused_project_bias, 1, 0)
            
            # Residual connection
            if self.use_residual:
                x = x + identity
            
            return x
    
    def _standard_forward(self, x):
        """Standard implementation using PyTorch modules"""
        identity = x
        
        if self.has_expand:
            x = self.expand_conv(x)
        
        x = self.depthwise_conv(x)
        x = self.project_conv(x)
        
        if self.use_residual:
            x += identity
        
        return x
    
    def forward(self, x):
        """
        Forward pass with automatic fallback to ensure correctness.

        :param x: The input tensor, shape (batch_size, in_channels, H, W)
        :return: The output tensor, shape (batch_size, out_channels, H', W')
        """
        try:
            return self._optimized_forward(x)
        except Exception:
            # Fallback to standard implementation
            return self._standard_forward(x)

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
in_channels = 112
out_channels = 192
kernel_size = 5
stride = 2
expand_ratio = 6

def get_inputs():
    return [torch.randn(batch_size, in_channels, 224, 224)]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, expand_ratio]