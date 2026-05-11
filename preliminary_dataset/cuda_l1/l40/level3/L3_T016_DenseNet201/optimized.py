import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedDenseBlockFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bn_weights, bn_biases, bn_running_means, bn_running_vars, 
                conv_weights, num_layers, num_input_features, growth_rate):
        batch_size, _, height, width = x.shape
        
        # Pre-allocate output tensor for all concatenated features
        total_features = num_input_features + num_layers * growth_rate
        output = torch.empty(batch_size, total_features, height, width, 
                            dtype=x.dtype, device=x.device)
        
        # Copy initial input features using narrow() for zero-copy tensor slicing
        output.narrow(1, 0, num_input_features).copy_(x)
        
        current_features = num_input_features
        
        # Process each layer with optimized memory access patterns
        for i in range(num_layers):
            # Use narrow() for efficient tensor slicing without creating new tensors
            layer_input = output.narrow(1, 0, current_features)
            
            # Extract parameters for current layer
            bn_weight = bn_weights[i]
            bn_bias = bn_biases[i]
            bn_mean = bn_running_means[i]
            bn_var = bn_running_vars[i]
            conv_weight = conv_weights[i]
            
            # BatchNorm operation
            bn_output = F.batch_norm(
                layer_input, 
                bn_mean, 
                bn_var, 
                bn_weight, 
                bn_bias,
                training=False,
                momentum=0.1,
                eps=1e-5
            )
            
            # In-place ReLU for memory efficiency
            F.relu_(bn_output)
            
            # Convolution
            conv_output = F.conv2d(bn_output, conv_weight, bias=None, stride=1, padding=1)
            
            # Direct memory copy to pre-allocated location
            output.narrow(1, current_features, growth_rate).copy_(conv_output)
            current_features += growth_rate
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Not needed for inference
        return None, None, None, None, None, None, None, None, None

class OptimizedDenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(OptimizedDenseBlock, self).__init__()
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.num_input_features = num_input_features
        
        # Create BatchNorm and Conv layers
        self.bn_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            self.bn_layers.append(nn.BatchNorm2d(in_features))
            self.conv_layers.append(nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False))
        
        # Dropout layer with 0.0 probability to match reference implementation
        self.dropout = nn.Dropout(0.0)
    
    def forward(self, x):
        bn_weights = []
        bn_biases = []
        bn_running_means = []
        bn_running_vars = []
        conv_weights = []
        
        for i in range(self.num_layers):
            bn_layer = self.bn_layers[i]
            conv_layer = self.conv_layers[i]
            
            bn_weights.append(bn_layer.weight)
            bn_biases.append(bn_layer.bias)
            bn_running_means.append(bn_layer.running_mean)
            bn_running_vars.append(bn_layer.running_var)
            conv_weights.append(conv_layer.weight)
        
        # Use our optimized implementation
        return OptimizedDenseBlockFunction.apply(
            x, bn_weights, bn_biases, bn_running_means, bn_running_vars, conv_weights,
            self.num_layers, self.num_input_features, self.growth_rate
        )

class FusedTransitionFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bn_weight, bn_bias, bn_mean, bn_var, conv_weight):
        # BatchNorm
        x = F.batch_norm(
            x, 
            bn_mean, 
            bn_var, 
            bn_weight, 
            bn_bias,
            training=False,
            momentum=0.1,
            eps=1e-5
        )
        
        # ReLU in-place
        F.relu_(x)
        
        # Convolution
        x = F.conv2d(x, conv_weight, bias=None)
        
        # Average pooling
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        # Not needed for inference
        return None, None, None, None, None, None

class OptimizedTransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(OptimizedTransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
    
    def forward(self, x):
        bn_weight = self.bn.weight
        bn_bias = self.bn.bias
        bn_mean = self.bn.running_mean
        bn_var = self.bn.running_var
        conv_weight = self.conv.weight
        
        # Use fused implementation
        return FusedTransitionFunction.apply(x, bn_weight, bn_bias, bn_mean, bn_var, conv_weight)

class FusedInitialFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, conv_weight, bn_weight, bn_bias, bn_mean, bn_var):
        # Convolution
        x = F.conv2d(x, conv_weight, bias=None, stride=2, padding=3)
        
        # BatchNorm
        x = F.batch_norm(
            x, 
            bn_mean, 
            bn_var, 
            bn_weight, 
            bn_bias,
            training=False,
            momentum=0.1,
            eps=1e-5
        )
        
        # ReLU in-place
        F.relu_(x)
        
        # MaxPool
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        # Not needed for inference
        return None, None, None, None, None, None

class FusedFinalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, bn_weight, bn_bias, bn_mean, bn_var, classifier_weight, classifier_bias):
        # BatchNorm
        x = F.batch_norm(
            x, 
            bn_mean, 
            bn_var, 
            bn_weight, 
            bn_bias,
            training=False,
            momentum=0.1,
            eps=1e-5
        )
        
        # ReLU in-place
        F.relu_(x)
        
        # Adaptive average pooling and flatten
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        # Linear classifier
        x = F.linear(x, classifier_weight, classifier_bias)
        
        return x
    
    @staticmethod
    def backward(ctx, grad_output):
        # Not needed for inference
        return None, None, None, None, None, None, None

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()
        
        # Initial convolution and pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dense blocks with our optimized implementation
        num_features = 64
        block_layers = [6, 12, 48, 32]  # DenseNet201 configuration
        
        self.dense_blocks = nn.ModuleList()
        self.transition_layers = nn.ModuleList()
        
        for i, num_layers in enumerate(block_layers):
            block = OptimizedDenseBlock(
                num_layers=num_layers, 
                num_input_features=num_features, 
                growth_rate=growth_rate
            )
            self.dense_blocks.append(block)
            num_features = num_features + num_layers * growth_rate
            
            if i != len(block_layers) - 1:
                transition = OptimizedTransitionLayer(
                    num_input_features=num_features, 
                    num_output_features=num_features // 2
                )
                self.transition_layers.append(transition)
                num_features = num_features // 2
        
        # Final batch norm and classifier
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial layers with fused operations
        conv_weight = self.conv1.weight
        bn_weight = self.bn1.weight
        bn_bias = self.bn1.bias
        bn_mean = self.bn1.running_mean
        bn_var = self.bn1.running_var
        
        x = FusedInitialFunction.apply(x, conv_weight, bn_weight, bn_bias, bn_mean, bn_var)
        
        # Dense blocks and transition layers
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)
        
        # Final processing with fused operations
        bn_weight = self.final_bn.weight
        bn_bias = self.final_bn.bias
        bn_mean = self.final_bn.running_mean
        bn_var = self.final_bn.running_var
        classifier_weight = self.classifier.weight
        classifier_bias = self.classifier.bias
        
        x = FusedFinalFunction.apply(x, bn_weight, bn_bias, bn_mean, bn_var, classifier_weight, classifier_bias)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
num_classes = 10
height, width = 224, 224  # Standard input size for DenseNet

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]