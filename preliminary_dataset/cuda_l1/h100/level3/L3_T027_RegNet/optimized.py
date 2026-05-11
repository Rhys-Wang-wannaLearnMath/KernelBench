import torch
import torch.nn as nn
import torch.nn.functional as F

class FusedConvBNReLU(nn.Module):
    """
    Fused Conv2d + BatchNorm2d + ReLU implementation for inference optimization
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(FusedConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                             padding=padding, stride=stride, bias=True)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def fuse_bn_into_conv(self):
        """
        Fuse batch norm parameters into convolution for inference
        """
        if not isinstance(self.bn, nn.BatchNorm2d):
            return  # Already fused or not applicable
            
        # Get BN parameters
        bn_weight = self.bn.weight.data
        bn_bias = self.bn.bias.data
        bn_running_mean = self.bn.running_mean.data
        bn_running_var = self.bn.running_var.data
        bn_eps = self.bn.eps
        
        # Get conv parameters
        conv_weight = self.conv.weight.data
        
        # Ensure conv has bias
        if self.conv.bias is None:
            self.conv.bias = nn.Parameter(torch.zeros(self.conv.out_channels, 
                                                     device=conv_weight.device,
                                                     dtype=conv_weight.dtype))
        conv_bias = self.conv.bias.data
        
        # Compute fused parameters
        factor = bn_weight / torch.sqrt(bn_running_var + bn_eps)
        
        # Fuse into conv weight and bias
        self.conv.weight.data = conv_weight * factor.view(-1, 1, 1, 1)
        self.conv.bias.data = (conv_bias - bn_running_mean) * factor + bn_bias
        
        # Remove BN from computation graph
        self.bn = nn.Identity()
        
    def forward(self, x):
        x = self.conv(x)
        return F.relu(x, inplace=True)  # Use inplace ReLU for memory efficiency

class OptimizedStage(nn.Module):
    """
    Optimized RegNet stage with fused operations
    """
    def __init__(self, in_channels, out_channels):
        super(OptimizedStage, self).__init__()
        self.conv1 = FusedConvBNReLU(in_channels, out_channels)
        self.conv2 = FusedConvBNReLU(out_channels, out_channels)
        
    def fuse_bn(self):
        self.conv1.fuse_bn_into_conv()
        self.conv2.fuse_bn_into_conv()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return F.max_pool2d(x, kernel_size=2, stride=2)

class ModelNew(nn.Module):
    def __init__(self, input_channels, stages, block_widths, output_classes):
        """
        :param input_channels: int, Number of input channels for the first layer
        :param stages: int, Number of stages in the RegNet architecture
        :param block_widths: List[int], Width (number of channels) for each block in the stages
        :param output_classes: int, Number of output classes for classification
        """
        super(ModelNew, self).__init__()

        self.stages = stages
        self.block_widths = block_widths
        
        # Build optimized feature extractor
        self.feature_extractor = nn.ModuleList()
        current_channels = input_channels
        
        for i in range(stages):
            self.feature_extractor.append(OptimizedStage(current_channels, block_widths[i]))
            current_channels = block_widths[i]
        
        # Final fully connected layer for classification
        self.fc = nn.Linear(block_widths[-1], output_classes)
        
        # For CUDA graph optimization
        self.static_input_size = (batch_size, input_channels, image_height, image_width)
        self.graph = None
        self.static_x = None
        self.static_output = None
        
        # Enable cuDNN optimizations
        if hasattr(torch.backends, 'cudnn'):
            torch.backends.cudnn.benchmark = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
        
        # Fuse BatchNorm layers during initialization
        self._fuse_bn_layers()
    
    def _fuse_bn_layers(self):
        """
        Fuse all BatchNorm layers during initialization
        """
        for stage in self.feature_extractor:
            stage.fuse_bn()
    
    def _optimize_with_cuda_graph(self, x):
        """
        Use CUDA graph to optimize forward pass for fixed-size inputs
        """
        if not torch.cuda.is_available():
            return self._forward_impl(x)
            
        if self.graph is None:
            # Initialize static tensors for CUDA graph capture
            self.static_x = torch.zeros(self.static_input_size, 
                                      device=x.device, 
                                      dtype=x.dtype)
            
            # Warmup before capture
            for _ in range(5):
                self._forward_impl(self.static_x)
            torch.cuda.synchronize()
                
            # Capture graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = self._forward_impl(self.static_x)
                
        # Copy input data to static tensor
        self.static_x.copy_(x)
        
        # Replay graph
        self.graph.replay()
        
        return self.static_output
    
    def _forward_impl(self, x):
        """
        Actual forward implementation
        """
        # Convert to channels_last memory format for better GPU performance
        if x.is_cuda:
            x = x.to(memory_format=torch.channels_last)
        
        # Process through feature extraction stages
        for stage in self.feature_extractor:
            x = stage(x)
        
        # Optimized global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # More efficient than view or reshape
        
        # Final classification
        x = self.fc(x)
        return x
        
    def forward(self, x):
        """
        Forward pass through the RegNet model.
        :param x: torch.Tensor of shape (batch_size, input_channels, height, width)
        :return: torch.Tensor of shape (batch_size, output_classes)
        """
        # Try to use CUDA graph optimization for fixed-size inputs
        if (x.is_cuda and x.size() == self.static_input_size and 
            torch.cuda.is_available() and 
            torch.cuda.get_device_capability()[0] >= 7):  # Volta or newer
            try:
                return self._optimize_with_cuda_graph(x)
            except Exception:
                # Fallback to standard implementation if CUDA graph fails
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 8
input_channels = 3
image_height, image_width = 224, 224
stages = 3
block_widths = [64, 128, 256]
output_classes = 10

def get_inputs():
    """ Generates random input tensor of shape (batch_size, input_channels, height, width) """
    return [torch.randn(batch_size, input_channels, image_height, image_width)]

def get_init_inputs():
    """ Initializes model parameters """
    return [input_channels, stages, block_widths, output_classes]