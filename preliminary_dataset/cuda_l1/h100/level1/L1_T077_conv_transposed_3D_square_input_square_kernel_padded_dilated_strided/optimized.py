import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelNew(nn.Module):
    """
    Performs a 3D transposed convolution operation with square input and square kernel,
    and supports padding, dilation, and stride.

    Args:
        in_channels (int): Number of channels in the input tensor.
        out_channels (int): Number of channels produced by the convolution.
        kernel_size (int): Size of the convolution kernel (square kernel, so only one value needed).
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding applied to the input. Defaults to 0.
        dilation (int, optional): Spacing between kernel elements. Defaults to 1.
        bias (bool, optional): If `True`, adds a learnable bias to the output. Defaults to `False`.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1, bias: bool = False):
        super(ModelNew, self).__init__()
        # Create the transposed convolution layer with the same parameters as the reference
        self.conv_transpose3d = nn.ConvTranspose3d(
            in_channels, out_channels, 
            kernel_size=(kernel_size, kernel_size, kernel_size), 
            stride=stride, padding=padding, dilation=dilation, bias=bias
        )
        
        # Enable cuDNN benchmarking for optimal algorithm selection
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Store parameters for direct function calls
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.bias = bias
        
        # Flags and caches for optimization
        self.optimized = False
        self.use_channels_last = False
        self.warmed_up = False
        
        # Cache for weights in different memory formats
        self._weight_standard = None
        self._weight_channels_last = None
        
        # Try to use TorchScript for additional optimization
        try:
            self._optimized_forward_fn = torch.jit.script(self._optimized_forward)
            self.use_script = True
        except Exception:
            self.use_script = False

    def _optimize_memory_format(self, x):
        """Determine if channels_last_3d format is beneficial and apply it if so"""
        with torch.no_grad():
            # Clone input for testing
            x_clone = x.clone()
            
            # Cache standard weight format
            self._weight_standard = self.conv_transpose3d.weight.detach().clone()
            if not self._weight_standard.is_contiguous():
                self._weight_standard = self._weight_standard.contiguous()
            
            # Test standard format
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            # Warm up standard format
            for _ in range(10):
                _ = self.conv_transpose3d(x_clone)
            torch.cuda.synchronize()
            
            # Benchmark standard format
            start.record()
            for _ in range(20):  # More iterations for better measurement
                _ = self.conv_transpose3d(x_clone)
            end.record()
            torch.cuda.synchronize()
            time_standard = start.elapsed_time(end)
            
            # Test channels_last_3d format if available
            if hasattr(torch, 'channels_last_3d'):
                try:
                    x_channels_last = x_clone.to(memory_format=torch.channels_last_3d)
                    weight_channels_last = self.conv_transpose3d.weight.to(memory_format=torch.channels_last_3d)
                    self._weight_channels_last = weight_channels_last.detach().clone()
                    
                    # Temporarily set weight to channels_last format
                    self.conv_transpose3d.weight.data = weight_channels_last
                    
                    # Warm up channels_last format
                    for _ in range(10):
                        _ = self.conv_transpose3d(x_channels_last)
                    torch.cuda.synchronize()
                    
                    # Benchmark channels_last format
                    start.record()
                    for _ in range(20):  # More iterations for better measurement
                        _ = self.conv_transpose3d(x_channels_last)
                    end.record()
                    torch.cuda.synchronize()
                    time_channels_last = start.elapsed_time(end)
                    
                    # Determine which format is faster
                    self.use_channels_last = time_channels_last < time_standard
                    
                    # Set weight to the optimal format
                    if self.use_channels_last:
                        self.conv_transpose3d.weight.data = self._weight_channels_last
                    else:
                        self.conv_transpose3d.weight.data = self._weight_standard
                except Exception:
                    # If channels_last_3d format causes issues, stick with standard format
                    self.use_channels_last = False
                    self.conv_transpose3d.weight.data = self._weight_standard
            else:
                # If channels_last_3d is not available, ensure weight is contiguous
                self.conv_transpose3d.weight.data = self._weight_standard

    def _optimized_forward(self, x):
        """Optimized forward implementation that can be JIT compiled"""
        # Apply memory format if beneficial
        if self.use_channels_last and hasattr(torch, 'channels_last_3d'):
            x = x.contiguous(memory_format=torch.channels_last_3d)
            weight = self._weight_channels_last if self._weight_channels_last is not None else self.conv_transpose3d.weight
        else:
            x = x.contiguous()
            weight = self._weight_standard if self._weight_standard is not None else self.conv_transpose3d.weight
        
        # Perform the transposed convolution directly with F.conv_transpose3d
        return F.conv_transpose3d(
            x,
            weight,
            self.conv_transpose3d.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.conv_transpose3d.output_padding,
            groups=self.conv_transpose3d.groups,
            dilation=self.dilation
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the 3D transposed convolution.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_channels, depth, height, width).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_channels, depth_out, height_out, width_out).
        """
        # Optimize memory format if not already done
        if not self.optimized and x.is_cuda:
            self._optimize_memory_format(x)
            self.optimized = True
        
        # Pre-warm cuDNN algorithms if not already done
        if not self.warmed_up and x.is_cuda:
            with torch.no_grad():
                # Apply memory format if beneficial
                if self.use_channels_last and hasattr(torch, 'channels_last_3d'):
                    x_warm = x.contiguous(memory_format=torch.channels_last_3d)
                else:
                    x_warm = x.contiguous()
                
                # Run multiple times to ensure cuDNN has selected optimal algorithm
                for _ in range(10):
                    _ = self._optimized_forward(x_warm)
                torch.cuda.synchronize()
            
            self.warmed_up = True
        
        # Use scripted forward if available
        if hasattr(self, 'use_script') and self.use_script:
            try:
                return self._optimized_forward_fn(x)
            except Exception:
                self.use_script = False
                return self._optimized_forward(x)
        
        # Use optimized forward directly if scripting failed
        return self._optimized_forward(x)

# Test code
batch_size = 16
in_channels = 32
out_channels = 64
kernel_size = 3
depth = 16
height = 32
width = 32
stride = 2
padding = 1
dilation = 2

def get_inputs():
    x = torch.randn(batch_size, in_channels, depth, height, width)
    return [x]

def get_init_inputs():
    return [in_channels, out_channels, kernel_size, stride, padding, dilation]