import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    def __init__(self, num_classes=1000):
        """
        Initialize the VGG16 model with optimized operations.
        
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(ModelNew, self).__init__()
        
        # Enable cuDNN benchmarking for automatic algorithm selection
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # VGG16 architecture: 5 blocks of convolutional layers followed by max pooling
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.0),
            nn.Linear(4096, num_classes)
        )
        
        # Check if GPU supports half precision (Tensor Cores)
        self.use_half = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 7
        
        # Pre-convert model to half precision if supported
        if self.use_half and torch.cuda.is_available():
            self.half()
        
        # JIT compile the features and classifier for better performance
        if torch.cuda.is_available():
            try:
                self.features = torch.jit.script(self.features)
                self.classifier = torch.jit.script(self.classifier)
            except Exception:
                pass
        
        # Convert model to channels_last memory format for better performance on NVIDIA GPUs
        if torch.cuda.is_available():
            try:
                self.to(memory_format=torch.channels_last)
            except Exception:
                pass
    
    def forward(self, x):
        """
        Forward pass of the VGG16 model with optimized execution.
        
        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        # Store original dtype for later conversion back if needed
        original_dtype = x.dtype
        
        # Convert input to channels_last for better performance
        if torch.cuda.is_available():
            x = x.contiguous(memory_format=torch.channels_last)
        
        # Use half precision if supported
        if self.use_half and x.is_cuda:
            x = x.half()
            
            # Process through convolutional layers
            x = self.features(x)
            
            # Flatten for fully connected layers
            x = torch.flatten(x, 1)
            
            # Process through classifier
            x = self.classifier(x)
        else:
            # Process through convolutional layers
            x = self.features(x)
            
            # Flatten for fully connected layers
            x = torch.flatten(x, 1)
            
            # Process through classifier
            x = self.classifier(x)
        
        # Ensure output has the same dtype as input
        if x.dtype != original_dtype:
            x = x.to(original_dtype)
            
        return x

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]