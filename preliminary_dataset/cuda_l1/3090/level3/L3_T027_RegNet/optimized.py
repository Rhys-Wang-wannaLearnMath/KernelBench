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
        self.fused = False
        
    def _fuse_bn_into_conv(self):
        """
        Fuse batch norm parameters into convolution for inference
        """
        if self.fused or not hasattr(self.bn, 'running_mean'):
            return
            
        with torch.no_grad():
            # Get BN parameters - ensure they're properly initialized
            if self.bn.running_mean is None:
                return  # BN not initialized yet
                
            bn_weight = self.bn.weight.data
            bn_bias = self.bn.bias.data
            bn_running_mean = self.bn.running_mean.data
            bn_running_var = self.bn.running_var.data
            bn_eps = self.bn.eps
            
            # Get conv parameters
            conv_weight = self.conv.weight.data
            
            # Ensure conv has bias (create if needed)
            if self.conv.bias is None:
                self.conv.bias = nn.Parameter(torch.zeros(self.conv.out_channels, 
                                                         device=conv_weight.device,
                                                         dtype=conv_weight.dtype))
            conv_bias = self.conv.bias.data
            
            # Compute fused parameters with numerical stability
            inv_std = torch.rsqrt(bn_running_var + bn_eps)
            scale = bn_weight * inv_std
            
            # Apply fusion
            self.conv.weight.data = conv_weight * scale.view(-1, 1, 1, 1)
            self.conv.bias.data = (conv_bias - bn_running_mean) * scale + bn_bias
            
            # Replace BN with identity and mark as fused
            self.bn = nn.Identity()
            self.fused = True
    
    def forward(self, x):
        """Optimized forward with lazy fusion"""
        # Attempt fusion on each forward pass until successful
        if not self.fused:
            self._fuse_bn_into_conv()
        
        # Forward pass
        x = self.conv(x)
        if not self.fused:
            x = self.bn(x)
        return F.relu(x, inplace=True)

class OptimizedStage(nn.Module):
    """
    Optimized RegNet stage with memory and computation optimizations
    """
    def __init__(self, in_channels, out_channels):
        super(OptimizedStage, self).__init__()
        self.conv1 = FusedConvBNReLU(in_channels, out_channels)
        self.conv2 = FusedConvBNReLU(out_channels, out_channels)
        
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
        
        # Build feature extractor with optimized stages
        self.feature_stages = nn.ModuleList()
        current_channels = input_channels
        
        for i in range(stages):
            stage = OptimizedStage(current_channels, block_widths[i])
            self.feature_stages.append(stage)
            current_channels = block_widths[i]
        
        # Final classification layer
        self.fc = nn.Linear(block_widths[-1], output_classes)
        
        # For CUDA graph optimization
        self.static_input_size = (batch_size, input_channels, image_height, image_width)
        self.graph = None
        self.static_x = None
        self.static_output = None
        self.warmup_complete = False
        
        # Apply global optimizations
        self._setup_optimizations()
    
    def _setup_optimizations(self):
        """Setup global optimizations for maximum performance"""
        # Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
    def _optimize_with_cuda_graph(self, x):
        """
        Use CUDA graph to optimize forward pass for fixed-size inputs
        """
        if not torch.cuda.is_available():
            return self._forward_impl(x)
            
        if self.graph is None:
            try:
                # Initialize static tensors for CUDA graph capture
                self.static_x = torch.zeros(self.static_input_size, 
                                          device=x.device, 
                                          dtype=x.dtype)
                
                # Warmup before capture to ensure all lazy initializations are done
                if not self.warmup_complete:
                    for _ in range(5):  # Increased warmup iterations for stability
                        self._forward_impl(self.static_x)
                    self.warmup_complete = True
                    
                # Capture graph
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.static_output = self._forward_impl(self.static_x)
            except Exception as e:
                # Fall back to regular execution if graph capture fails
                self.graph = None
                return self._forward_impl(x)
                
        # Copy input data to static tensor
        self.static_x.copy_(x)
        
        # Replay graph
        self.graph.replay()
        
        # Return output
        return self.static_output
    
    def _forward_impl(self, x):
        """
        Actual forward implementation
        """
        # Convert to channels_last memory format for better GPU performance
        if x.is_cuda and x.dim() == 4 and not x.is_contiguous(memory_format=torch.channels_last):
            x = x.to(memory_format=torch.channels_last)
        
        # Process through feature extraction stages
        for stage in self.feature_stages:
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
            return self._optimize_with_cuda_graph(x)
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