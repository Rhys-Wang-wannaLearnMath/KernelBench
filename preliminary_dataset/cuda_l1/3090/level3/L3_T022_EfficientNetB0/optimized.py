import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedMBConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio):
        """
        Optimized MBConv block implementation with batch normalization fusion.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Kernel size for the depthwise convolution.
            stride: Stride for the depthwise convolution.
            expand_ratio: Expansion ratio for the intermediate channels.
        """
        super(OptimizedMBConv, self).__init__()
        
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.hidden_dim = in_channels * expand_ratio
        self.expand_ratio = expand_ratio
        
        # Expand phase
        if expand_ratio != 1:
            self.expand_conv = nn.Conv2d(in_channels, self.hidden_dim, kernel_size=1, stride=1, padding=0, bias=False)
            self.expand_bn = nn.BatchNorm2d(self.hidden_dim)
        
        # Depthwise phase
        self.depthwise_conv = nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=kernel_size, stride=stride, 
                                      padding=(kernel_size-1)//2, groups=self.hidden_dim, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(self.hidden_dim)
        
        # Project phase
        self.project_conv = nn.Conv2d(self.hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.project_bn = nn.BatchNorm2d(out_channels)
        
        # For fused operations in inference mode
        self.fused_expand = None
        self.fused_depthwise = None
        self.fused_project = None
    
    def _fuse_bn_tensor(self, conv, bn):
        """
        Fuse batch normalization into convolution weights for inference.
        
        Args:
            conv: Convolution layer
            bn: Batch normalization layer
            
        Returns:
            Tuple of (fused_weight, fused_bias)
        """
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        return kernel * t, beta - running_mean * gamma / std
    
    def _fuse_operations(self):
        """
        Fuse batch normalization into convolution for faster inference.
        """
        if self.fused_project is not None:  # Already fused
            return
            
        if self.expand_ratio != 1:
            w1, b1 = self._fuse_bn_tensor(self.expand_conv, self.expand_bn)
            self.fused_expand = nn.Conv2d(
                self.expand_conv.in_channels, self.expand_conv.out_channels,
                kernel_size=self.expand_conv.kernel_size, stride=self.expand_conv.stride,
                padding=self.expand_conv.padding, bias=True
            )
            self.fused_expand.weight.data = w1
            self.fused_expand.bias.data = b1
        
        w2, b2 = self._fuse_bn_tensor(self.depthwise_conv, self.depthwise_bn)
        self.fused_depthwise = nn.Conv2d(
            self.depthwise_conv.in_channels, self.depthwise_conv.out_channels,
            kernel_size=self.depthwise_conv.kernel_size, stride=self.depthwise_conv.stride,
            padding=self.depthwise_conv.padding, groups=self.depthwise_conv.groups, bias=True
        )
        self.fused_depthwise.weight.data = w2
        self.fused_depthwise.bias.data = b2
        
        w3, b3 = self._fuse_bn_tensor(self.project_conv, self.project_bn)
        self.fused_project = nn.Conv2d(
            self.project_conv.in_channels, self.project_conv.out_channels,
            kernel_size=self.project_conv.kernel_size, stride=self.project_conv.stride,
            padding=self.project_conv.padding, bias=True
        )
        self.fused_project.weight.data = w3
        self.fused_project.bias.data = b3
    
    def forward(self, x):
        """
        Forward pass of the optimized MBConv block.
        
        Args:
            x: The input tensor
            
        Returns:
            The output tensor
        """
        # Store residual at the beginning if needed
        if self.use_residual:
            identity = x
        
        # Optimized inference path with fused operations
        if not self.training and self.fused_project is not None:
            # Expand phase
            if self.expand_ratio != 1:
                x = F.relu6(self.fused_expand(x), inplace=True)
            
            # Depthwise phase
            x = F.relu6(self.fused_depthwise(x), inplace=True)
            
            # Project phase
            x = self.fused_project(x)
            
            # Residual connection
            if self.use_residual:
                x = x + identity
            
            return x
        
        # Standard training path
        else:
            # Expand phase
            if self.expand_ratio != 1:
                x = self.expand_conv(x)
                x = self.expand_bn(x)
                x = F.relu6(x, inplace=True)
            
            # Depthwise phase
            x = self.depthwise_conv(x)
            x = self.depthwise_bn(x)
            x = F.relu6(x, inplace=True)
            
            # Project phase
            x = self.project_conv(x)
            x = self.project_bn(x)
            
            # Residual connection
            if self.use_residual:
                x = x + identity
            
            return x

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Optimized EfficientNetB0 architecture implementation.

        Args:
            num_classes: The number of output classes (default is 1000 for ImageNet).
        """
        super(ModelNew, self).__init__()
        
        # Initial convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        
        # MBConv blocks with optimized implementation
        self.blocks = nn.ModuleList([
            # MBConv1 (32, 16, 1, 1)
            OptimizedMBConv(32, 16, kernel_size=3, stride=1, expand_ratio=1),
            # MBConv6 (16, 24, 2, 6)
            OptimizedMBConv(16, 24, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (24, 24, 1, 6)
            OptimizedMBConv(24, 24, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (24, 40, 2, 6)
            OptimizedMBConv(24, 40, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (40, 40, 1, 6)
            OptimizedMBConv(40, 40, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (40, 80, 2, 6)
            OptimizedMBConv(40, 80, kernel_size=3, stride=2, expand_ratio=6),
            # MBConv6 (80, 80, 1, 6)
            OptimizedMBConv(80, 80, kernel_size=3, stride=1, expand_ratio=6),
            # MBConv6 (80, 112, 1, 6)
            OptimizedMBConv(80, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 112, 1, 6)
            OptimizedMBConv(112, 112, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (112, 192, 2, 6)
            OptimizedMBConv(112, 192, kernel_size=5, stride=2, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            OptimizedMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 192, 1, 6)
            OptimizedMBConv(192, 192, kernel_size=5, stride=1, expand_ratio=6),
            # MBConv6 (192, 320, 1, 6)
            OptimizedMBConv(192, 320, kernel_size=3, stride=1, expand_ratio=6)
        ])
        
        # Final convolutional layer
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        
        # Fully connected layer
        self.fc = nn.Linear(1280, num_classes)
        
        # For fused operations in inference mode
        self.fused_conv1 = None
        self.fused_conv2 = None
        
        # For memory format optimization
        self.use_channels_last = False
        
        # Apply optimization techniques if CUDA is available
        if torch.cuda.is_available():
            self._optimize_model()
    
    def _fuse_bn_tensor(self, conv, bn):
        """
        Fuse batch normalization into convolution weights for inference.
        
        Args:
            conv: Convolution layer
            bn: Batch normalization layer
            
        Returns:
            Tuple of (fused_weight, fused_bias)
        """
        kernel = conv.weight
        running_mean = bn.running_mean
        running_var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        
        return kernel * t, beta - running_mean * gamma / std
    
    def _fuse_operations(self):
        """
        Fuse batch normalization into convolution for faster inference.
        """
        # Fuse initial conv+bn
        w1, b1 = self._fuse_bn_tensor(self.conv1, self.bn1)
        self.fused_conv1 = nn.Conv2d(
            self.conv1.in_channels, self.conv1.out_channels,
            kernel_size=self.conv1.kernel_size, stride=self.conv1.stride,
            padding=self.conv1.padding, bias=True
        )
        self.fused_conv1.weight.data = w1
        self.fused_conv1.bias.data = b1
        
        # Fuse final conv+bn
        w2, b2 = self._fuse_bn_tensor(self.conv2, self.bn2)
        self.fused_conv2 = nn.Conv2d(
            self.conv2.in_channels, self.conv2.out_channels,
            kernel_size=self.conv2.kernel_size, stride=self.conv2.stride,
            padding=self.conv2.padding, bias=True
        )
        self.fused_conv2.weight.data = w2
        self.fused_conv2.bias.data = b2
        
        # Pre-fuse all MBConv blocks
        for block in self.blocks:
            block._fuse_operations()
    
    def _optimize_model(self):
        """
        Apply optimization techniques to the model.
        """
        # Ensure the model is in eval mode for optimization
        self.eval()
        
        # Try to enable more aggressive JIT fusion
        try:
            torch._C._jit_set_profiling_mode(False)
            torch._C._jit_set_bailout_depth(20)
            torch._C._jit_override_can_fuse_on_cpu(True)
            torch._C._jit_override_can_fuse_on_gpu(True)
        except:
            pass
        
        # Enable channels_last memory format for better performance on CUDA
        self.use_channels_last = True
        self = self.to(memory_format=torch.channels_last)
        
        # Pre-fuse operations for faster first inference
        self._fuse_operations()
    
    def _forward_impl(self, x):
        """
        Implementation of the forward pass without CUDA graph.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Initial convolution
        if not self.training and self.fused_conv1 is not None:
            x = F.relu(self.fused_conv1(x), inplace=True)
        else:
            x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        
        # MBConv blocks
        for block in self.blocks:
            x = block(x)
        
        # Final convolution
        if not self.training and self.fused_conv2 is not None:
            x = F.relu(self.fused_conv2(x), inplace=True)
        else:
            x = F.relu(self.bn2(self.conv2(x)), inplace=True)
        
        # Global average pooling and classification
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
    def forward(self, x):
        """
        Forward pass of the optimized EfficientNetB0 model.
        
        Args:
            x: The input tensor, shape (batch_size, 3, 224, 224)
            
        Returns:
            The output tensor, shape (batch_size, num_classes)
        """
        # Convert to channels_last format if enabled and on CUDA
        if self.use_channels_last and x.device.type == 'cuda':
            x = x.contiguous(memory_format=torch.channels_last)
        
        # Fall back to regular forward pass
        return self._forward_impl(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]