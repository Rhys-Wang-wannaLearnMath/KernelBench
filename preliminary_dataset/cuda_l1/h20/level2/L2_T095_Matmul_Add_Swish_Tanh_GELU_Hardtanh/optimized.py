import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        in_features (int): Number of input features
        out_features (int): Number of output features  
        add_value_shape (tuple): Shape of the add_value parameter
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(ModelNew, self).__init__()
        # Use nn.Linear to ensure identical weight/bias initialization
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        
        # Direct references to parameters for more efficient access
        self.weight = self.matmul.weight
        self.bias = self.matmul.bias
        
        # Pre-compute optimized tensors with explicit memory layout
        self.register_buffer('weight_t', self.weight.t().contiguous())
        self.register_buffer('combined_bias', (self.bias + self.add_value).contiguous())
        
        # Flag for evaluation mode update
        self.register_buffer('eval_update_needed', torch.tensor(True))
        
        # Register minimal parameter hooks
        self._register_minimal_hooks()
    
    def _register_minimal_hooks(self):
        """Register minimal hooks to update buffers when parameters change"""
        def update_weight_buffer(grad):
            if grad is not None and self.training:
                with torch.no_grad():
                    self.weight_t.copy_(self.weight.t().contiguous())
                    # Reset eval flag when weights change during training
                    self.eval_update_needed.fill_(True)
        
        def update_bias_buffer(grad):
            if grad is not None and self.training:
                with torch.no_grad():
                    self.combined_bias.copy_((self.bias + self.add_value).contiguous())
                    # Reset eval flag when biases change during training
                    self.eval_update_needed.fill_(True)
        
        # Register parameter-level hooks for minimal overhead
        self.weight.register_hook(update_weight_buffer)
        self.bias.register_hook(update_bias_buffer)
        self.add_value.register_hook(update_bias_buffer)
    
    def forward(self, x):
        """
        Ultra-optimized forward pass with zero conditional logic
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Update buffers in eval mode if needed (only once after switching from training)
        if not self.training and self.eval_update_needed:
            with torch.no_grad():
                self.weight_t.copy_(self.weight.t().contiguous())
                self.combined_bias.copy_((self.bias + self.add_value).contiguous())
                self.eval_update_needed.fill_(False)
        
        # Fused matrix multiplication with bias addition
        # addmm: out = beta * input + alpha * (mat1 @ mat2)
        x = torch.addmm(self.combined_bias, x, self.weight_t)
        
        # Optimized Swish activation: x * sigmoid(x)
        sigmoid_x = torch.sigmoid(x)
        x.mul_(sigmoid_x)  # In-place multiplication
        
        # Apply remaining activations in sequence
        x = torch.tanh(x)
        x = torch.nn.functional.gelu(x)
        
        # In-place hardtanh (clamp between -1 and 1)
        x.clamp_(-1.0, 1.0)
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
in_features = 1024
out_features = 512
add_value_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]