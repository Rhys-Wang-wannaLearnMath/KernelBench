import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.cuda.amp as amp

class FusedConvReLU(nn.Module):
    """
    Fused Conv2d + ReLU module for better performance
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FusedConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        return F.relu(self.conv(x), inplace=True)

class FusedConvReLUPool(nn.Module):
    """
    Fused Conv2d + ReLU + MaxPool module for better performance
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, 
                 pool_kernel_size=3, pool_stride=2):
        super(FusedConvReLUPool, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.pool_kernel_size = pool_kernel_size
        self.pool_stride = pool_stride
        
    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x, inplace=True)
        return F.max_pool2d(x, self.pool_kernel_size, self.pool_stride)

class OptimizedLinearReLU(nn.Module):
    """
    Optimized Linear + ReLU + Dropout module
    """
    def __init__(self, in_features, out_features, dropout_prob=0.0):
        super(OptimizedLinearReLU, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.dropout_prob = dropout_prob
        
    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x, inplace=True)
        if self.dropout_prob > 0 and self.training:
            x = F.dropout(x, p=self.dropout_prob, training=self.training, inplace=True)
        return x

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(ModelNew, self).__init__()
        
        # Use mixed precision for better performance on compatible GPUs
        self.use_amp = torch.cuda.is_available()
        
        # First convolutional layer with maxpool - fused operations
        self.features1 = FusedConvReLUPool(
            in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2,
            pool_kernel_size=3, pool_stride=2
        )
        
        # Second convolutional layer with maxpool - fused operations
        self.features2 = FusedConvReLUPool(
            in_channels=96, out_channels=256, kernel_size=5, padding=2,
            pool_kernel_size=3, pool_stride=2
        )
        
        # Third convolutional layer - fused Conv+ReLU
        self.features3 = FusedConvReLU(
            in_channels=256, out_channels=384, kernel_size=3, padding=1
        )
        
        # Fourth convolutional layer - fused Conv+ReLU
        self.features4 = FusedConvReLU(
            in_channels=384, out_channels=384, kernel_size=3, padding=1
        )
        
        # Fifth convolutional layer with maxpool - separate operations for better optimization
        self.features5 = FusedConvReLU(
            in_channels=384, out_channels=256, kernel_size=3, padding=1
        )
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers with fused operations
        self.fc1 = OptimizedLinearReLU(
            in_features=256 * 6 * 6, out_features=4096, dropout_prob=0.0
        )
        
        self.fc2 = OptimizedLinearReLU(
            in_features=4096, out_features=4096, dropout_prob=0.0
        )
        
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # Use mixed precision for better performance
        if self.use_amp and self.training:
            with amp.autocast():
                return self._forward_impl(x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x):
        # Ensure input is contiguous for better memory access patterns
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Convolutional layers with fused operations
        x = self.features1(x)
        x = self.features2(x)
        x = self.features3(x)
        x = self.features4(x)
        x = self.features5(x)
        x = self.maxpool3(x)
        
        # Flatten - ensure contiguous memory for efficient linear layer computation
        x = torch.flatten(x, 1)
        
        # Fully connected layers with fused operations
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        
        return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]