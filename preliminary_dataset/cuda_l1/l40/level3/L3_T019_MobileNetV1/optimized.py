import torch
import torch.nn as nn
import torch.nn.functional as F

class FoldedConvBNReLU(nn.Module):
    """
    Module that folds BatchNorm into Conv2d for inference efficiency
    """
    def __init__(self, conv, bn):
        super(FoldedConvBNReLU, self).__init__()
        self.conv = conv
        self.bn = bn
        self.folded = False
        self.is_depthwise = conv.groups == conv.in_channels and conv.in_channels > 1
        
    def fold_bn(self):
        if self.folded:
            return
            
        # Get original weights and bias
        w = self.conv.weight
        b = torch.zeros(w.size(0), device=w.device) if self.conv.bias is None else self.conv.bias
        
        # Get BatchNorm parameters
        bn_w = self.bn.weight
        bn_b = self.bn.bias
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        bn_eps = self.bn.eps
        
        # Fold BatchNorm into Conv
        factor = bn_w / torch.sqrt(bn_var + bn_eps)
        
        # For depthwise conv, we need to reshape factor appropriately
        if self.is_depthwise:
            factor = factor.view(-1, 1, 1, 1)
        else:
            factor = factor.view(-1, 1, 1, 1)
            
        self.conv.weight.data = w * factor
        self.conv.bias = nn.Parameter(bn_b + (b - bn_mean) * factor.view(-1))
        
        self.folded = True
    
    def forward(self, x):
        if not self.training and not self.folded:
            self.fold_bn()
            
        return F.relu(self.conv(x), inplace=True)

class OptimizedDepthwiseSeparable(nn.Module):
    def __init__(self, inp, oup, stride):
        super(OptimizedDepthwiseSeparable, self).__init__()
        
        # Depthwise convolution with BatchNorm and ReLU
        self.depthwise_conv = nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False)
        self.depthwise_bn = nn.BatchNorm2d(inp)
        self.depthwise = FoldedConvBNReLU(self.depthwise_conv, self.depthwise_bn)
        
        # Pointwise convolution with BatchNorm and ReLU
        self.pointwise_conv = nn.Conv2d(inp, oup, 1, 1, 0, bias=False)
        self.pointwise_bn = nn.BatchNorm2d(oup)
        self.pointwise = FoldedConvBNReLU(self.pointwise_conv, self.pointwise_bn)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
        """
        MobileNetV1 architecture implementation.

        :param num_classes: The number of output classes (default: 1000)
        :param input_channels: The number of input channels (default: 3 for RGB images)
        :param alpha: Width multiplier (default: 1.0)
        """
        super(ModelNew, self).__init__()
        
        def conv_bn(inp, oup, stride):
            conv = nn.Conv2d(inp, oup, 3, stride, 1, bias=False)
            bn = nn.BatchNorm2d(oup)
            return FoldedConvBNReLU(conv, bn)
        
        def conv_dw(inp, oup, stride):
            return OptimizedDepthwiseSeparable(inp, oup, stride)
        
        # Follow the exact same structure as the reference implementation
        self.model = nn.Sequential(
            conv_bn(input_channels, int(32 * alpha), 2),
            conv_dw(int(32 * alpha), int(64 * alpha), 1),
            conv_dw(int(64 * alpha), int(128 * alpha), 2),
            conv_dw(int(128 * alpha), int(128 * alpha), 1),
            conv_dw(int(128 * alpha), int(256 * alpha), 2),
            conv_dw(int(256 * alpha), int(256 * alpha), 1),
            conv_dw(int(256 * alpha), int(512 * alpha), 2),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(512 * alpha), 1),
            conv_dw(int(512 * alpha), int(1024 * alpha), 2),
            conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(int(1024 * alpha), num_classes)
        
        # Enable optimizations
        self._enable_optimizations()
        
        # Pre-fold BatchNorm layers for inference if not in training mode
        if not self.training:
            self._fold_batchnorm()
        
    def _enable_optimizations(self):
        """Enable various PyTorch optimizations"""
        if torch.cuda.is_available():
            # Enable cuDNN benchmarking to find the best algorithm
            torch.backends.cudnn.benchmark = True
            
            # Enable TF32 precision for faster computation on Ampere+ GPUs
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            
            # Convert to channels_last format for better memory access
            self = self.to(memory_format=torch.channels_last)
    
    def _fold_batchnorm(self):
        """Fold BatchNorm into Conv layers for inference"""
        for module in self.modules():
            if isinstance(module, FoldedConvBNReLU) and not module.folded:
                module.fold_bn()
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_channels, height, width)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # Convert to channels_last format for better memory access if on CUDA
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last)
        
        # Apply model with optimized execution path
        if not self.training:
            with torch.no_grad():
                x = self.model(x)
                x = torch.flatten(x, 1)  # More efficient than view/reshape
                x = self.fc(x)
        else:
            x = self.model(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
input_channels = 3
height = 224
width = 224
num_classes = 1000
alpha = 1.0

def get_inputs():
    return [torch.randn(batch_size, input_channels, height, width)]

def get_init_inputs():
    return [num_classes, input_channels, alpha]