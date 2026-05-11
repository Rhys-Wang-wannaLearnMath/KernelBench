import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedDenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        super(OptimizedDenseBlock, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        
        # Create BatchNorm and Conv layers
        self.bn_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            self.bn_layers.append(nn.BatchNorm2d(in_features))
            self.conv_layers.append(nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False))
    
    def forward(self, x):
        batch_size, _, height, width = x.shape
        
        # Pre-allocate output tensor for all concatenated features
        total_features = self.num_input_features + self.num_layers * self.growth_rate
        output = torch.empty(batch_size, total_features, height, width, 
                            dtype=x.dtype, device=x.device)
        
        # Copy initial input features
        output.narrow(1, 0, self.num_input_features).copy_(x)
        
        current_features = self.num_input_features
        
        # Process each layer
        for i in range(self.num_layers):
            layer_input = output.narrow(1, 0, current_features)
            
            bn_layer = self.bn_layers[i]
            conv_layer = self.conv_layers[i]
            
            # BatchNorm
            bn_output = F.batch_norm(
                layer_input, 
                bn_layer.running_mean, 
                bn_layer.running_var, 
                bn_layer.weight, 
                bn_layer.bias,
                training=False,
                momentum=0.1,
                eps=1e-5
            )
            
            # In-place ReLU
            F.relu_(bn_output)
            
            # Convolution
            conv_output = F.conv2d(bn_output, conv_layer.weight, bias=None, stride=1, padding=1)
            
            # Copy to output tensor
            output.narrow(1, current_features, self.growth_rate).copy_(conv_output)
            current_features += self.growth_rate
        
        return output

class OptimizedTransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super(OptimizedTransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        # BatchNorm
        x = F.batch_norm(
            x, 
            self.bn.running_mean, 
            self.bn.running_var, 
            self.bn.weight, 
            self.bn.bias,
            training=False,
            momentum=0.1,
            eps=1e-5
        )
        
        # In-place ReLU
        F.relu_(x)
        
        # Convolution
        x = F.conv2d(x, self.conv.weight, bias=None)
        
        # Pooling
        x = self.pool(x)
        
        return x

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        super(ModelNew, self).__init__()
        
        # Initial convolution and pooling
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Dense blocks with optimized implementation
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
        # Initial convolution
        x = F.conv2d(x, self.conv1.weight, bias=None, stride=2, padding=3)
        
        # BatchNorm + ReLU
        x = F.batch_norm(
            x,
            self.bn1.running_mean,
            self.bn1.running_var,
            self.bn1.weight,
            self.bn1.bias,
            training=False,
            momentum=0.1,
            eps=1e-5
        )
        F.relu_(x)  # In-place ReLU
        
        # MaxPool
        x = self.maxpool(x)
        
        # Dense blocks and transition layers
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)
        
        # Final BatchNorm + ReLU
        x = F.batch_norm(
            x,
            self.final_bn.running_mean,
            self.final_bn.running_var,
            self.final_bn.weight,
            self.final_bn.bias,
            training=False,
            momentum=0.1,
            eps=1e-5
        )
        F.relu_(x)  # In-place ReLU
        
        # Global average pooling and classification
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
num_classes = 10
height, width = 224, 224  # Standard input size for DenseNet

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]