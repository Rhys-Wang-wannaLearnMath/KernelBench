import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Performs a standard 2D convolution operation with asymmetric input and kernel sizes.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (tuple): Tuple of two integers representing the height and width of the convolution kernel.
        stride (tuple, optional): Tuple of two integers representing the stride in the height and width dimensions. Defaults to (1, 1).
        padding (tuple, optional): Tuple of two integers representing the padding in the height and width dimensions. Defaults to (0, 0).
        dilation (tuple, optional): Tuple of two integers representing the dilation in the height and width dimensions. Defaults to (1, 1).
        groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple, 
                 stride: tuple = (1, 1), padding: tuple = (0, 0), 
                 dilation: tuple = (1, 1), groups: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        
        # Create standard PyTorch convolution layer
        self.conv2d = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, dilation=dilation,
            groups=groups, bias=bias
        )
        
        # Optimization flags
        self.use_channels_last = True  # Default to True as it's usually faster for convolutions
        
        # Warmup state
        self.warmup_complete = False
        
        # Save original flags to restore later
        self.original_flags = {}
        
        # Enable tensor cores for both cuDNN and CUDA if available
        if hasattr(torch.backends.cudnn, 'allow_tf32'):
            self.original_flags['allow_tf32'] = torch.backends.cudnn.allow_tf32
            torch.backends.cudnn.allow_tf32 = True
        
        if hasattr(torch.backends.cuda, 'matmul.allow_tf32'):
            self.original_flags['matmul_allow_tf32'] = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = True
        
        # Enable cuDNN benchmark mode for algorithm selection
        self.original_flags['benchmark'] = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = True
        
        # Increase workspace size limit for potentially faster algorithms
        if hasattr(torch.backends.cudnn, 'workspace_limit'):
            self.original_flags['workspace_limit'] = torch.backends.cudnn.workspace_limit
            torch.backends.cudnn.workspace_limit = 768 * 1024 * 1024  # 768 MB - optimized size
        
        # Pre-convert weights to channels_last format as it's usually faster
        if torch.cuda.is_available():
            self.conv2d.weight.data = self.conv2d.weight.data.to(memory_format=torch.channels_last)
    
    def __del__(self):
        # Restore original flags when the model is deleted
        for key, value in self.original_flags.items():
            if key == 'allow_tf32' and hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = value
            elif key == 'matmul_allow_tf32' and hasattr(torch.backends.cuda, 'matmul.allow_tf32'):
                torch.backends.cuda.matmul.allow_tf32 = value
            elif key == 'benchmark':
                torch.backends.cudnn.benchmark = value
            elif key == 'workspace_limit' and hasattr(torch.backends.cudnn, 'workspace_limit'):
                torch.backends.cudnn.workspace_limit = value
    
    def _should_use_channels_last(self, x):
        """
        Heuristic to determine if channels_last format would be beneficial
        without expensive benchmarking
        """
        # Skip if not on CUDA
        if not x.is_cuda:
            return False
            
        # For this specific convolution case (asymmetric kernel and input dimensions),
        # channels_last is almost always better
        return True
    
    def _run_warmup(self, x):
        """Ultra-efficient warmup to find optimal configuration"""
        if not x.is_cuda or self.warmup_complete:
            return
        
        # Use heuristic to determine memory format
        self.use_channels_last = self._should_use_channels_last(x)
        
        # Run a single iteration to let cuDNN find the best algorithm
        with torch.no_grad():
            if self.use_channels_last:
                x_opt = x.to(memory_format=torch.channels_last)
                _ = self.conv2d(x_opt)
        
        # Mark warmup as complete immediately to avoid repeated warmup
        self.warmup_complete = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 2D convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, height_out, width_out).
        """
        # Run warmup if needed
        if x.is_cuda and not self.warmup_complete:
            self._run_warmup(x)
        
        # Fast path for non-CUDA tensors
        if not x.is_cuda:
            return self.conv2d(x)
        
        # Use the optimal configuration determined during warmup
        if self.use_channels_last and x.dim() == 4:
            x = x.to(memory_format=torch.channels_last)
        
        # Execute the convolution
        output = self.conv2d(x)
        return output

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 16
in_channels = 3
out_channels = 64
kernel_size = (3, 5)  # Asymmetric kernel
height = 256
width = 128  # Asymmetric input dimensions

def get_inputs():
    x = torch.randn(batch_size, in_channels, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size]  # Provide in_channels, out_channels, kernel_size for initialization