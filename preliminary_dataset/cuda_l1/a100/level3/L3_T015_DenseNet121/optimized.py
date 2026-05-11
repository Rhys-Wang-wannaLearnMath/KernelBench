import torch
import torch.nn as nn
import torch.nn.functional as F

class OptimizedDenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int):
        """
        :param num_layers: The number of layers in the dense block
        :param num_input_features: The number of input feature maps
        :param growth_rate: The growth rate for the dense block (new features added per layer)
        """
        super(OptimizedDenseBlock, self).__init__()
        self.num_layers = num_layers
        self.num_input_features = num_input_features
        self.growth_rate = growth_rate
        
        # Create layers with the same structure as the reference implementation
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_features = num_input_features + i * growth_rate
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(in_features),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_features, growth_rate, kernel_size=3, padding=1, bias=False),
                nn.Dropout(0.0)
            ))
        
        # Pre-calculate the final number of features
        self.num_output_features = num_input_features + num_layers * growth_rate
        
        # Register buffer for feature storage with persistent=False to avoid saving in state_dict
        self.register_buffer('feature_buffer', None, persistent=False)
        self.last_input_shape = None

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Output tensor with shape (batch_size, num_output_features, height, width)
        """
        batch_size, _, height, width = x.shape
        device = x.device
        dtype = x.dtype
        current_shape = (batch_size, height, width)
        
        # Allocate or reuse feature buffer
        if (self.feature_buffer is None or 
            self.last_input_shape != current_shape or 
            self.feature_buffer.shape[0] != batch_size or
            self.feature_buffer.shape[2] != height or
            self.feature_buffer.shape[3] != width):
            
            # Use the same memory format as input for better performance
            memory_format = torch.channels_last if x.is_contiguous(memory_format=torch.channels_last) else torch.contiguous_format
            
            self.feature_buffer = torch.empty(
                batch_size, 
                self.num_output_features, 
                height, 
                width, 
                device=device, 
                dtype=dtype,
                memory_format=memory_format
            )
            self.last_input_shape = current_shape
        
        # Copy input features to the beginning of feature_buffer using narrow for efficiency
        self.feature_buffer.narrow(1, 0, self.num_input_features).copy_(x)
        
        # Process each layer and store results directly in feature_buffer
        features_so_far = self.num_input_features
        for i, layer in enumerate(self.layers):
            # Use narrow to create a view without allocating new memory
            current_input = self.feature_buffer.narrow(1, 0, features_so_far)
            
            # Process through the layer
            new_feature = layer(current_input)
            
            # Store new features directly in the buffer using narrow
            self.feature_buffer.narrow(1, features_so_far, self.growth_rate).copy_(new_feature)
            
            # Update the number of accumulated features for next layer
            features_so_far += self.growth_rate
        
        return self.feature_buffer

class TransitionLayer(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        return self.transition(x)

class ModelNew(nn.Module):
    def __init__(self, growth_rate: int = 32, num_classes: int = 1000):
        """
        :param growth_rate: The growth rate of the DenseNet (new features added per layer)
        :param num_classes: The number of output classes for classification
        """
        super(ModelNew, self).__init__()

        # Initial convolution and pooling
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Each dense block is followed by a transition layer, except the last one
        num_features = 64
        block_layers = [6, 12, 24, 16]  # Corresponding layers in DenseNet121

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
                transition = TransitionLayer(
                    num_input_features=num_features, 
                    num_output_features=num_features // 2
                )
                self.transition_layers.append(transition)
                num_features = num_features // 2

        # Final batch norm and classifier
        self.final_bn = nn.BatchNorm2d(num_features)
        self.classifier = nn.Linear(num_features, num_classes)
        
        # Enable performance optimizations
        if torch.cuda.is_available():
            # Enable cuDNN benchmark mode for consistent input sizes
            torch.backends.cudnn.benchmark = True
            
            # Enable TensorFloat-32 for faster computation on Ampere GPUs
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Input tensor of shape (batch_size, 3, height, width)
        :return: Output tensor of shape (batch_size, num_classes)
        """
        # Ensure input is contiguous for better performance
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Fall back to eager execution
        return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implementation of the forward pass
        
        :param x: Input tensor of shape (batch_size, 3, height, width)
        :return: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)
        
        for i, block in enumerate(self.dense_blocks):
            x = block(x)
            if i != len(self.dense_blocks) - 1:
                x = self.transition_layers[i](x)
        
        x = self.final_bn(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
num_classes = 10
height, width = 224, 224  # Standard input size for DenseNet

def get_inputs():
    return [torch.randn(batch_size, 3, height, width)]

def get_init_inputs():
    return [32, num_classes]