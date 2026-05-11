import torch
import torch.nn as nn

class ModelNew(nn.Module):
    """
    Optimized implementation that maintains identical functionality
    but with improved CUDA kernel performance
    
    Args:
        input_size (int): Number of input features
        hidden_size (int): Number of hidden features
        output_size (int): Number of output features
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(ModelNew, self).__init__()
        # Keep the original linear layers for parameter management
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        
        # Pre-transpose the weight matrix for more efficient matrix multiplication
        self.register_buffer('weight_t', self.linear1.weight.t().contiguous())
        self.register_buffer('bias', self.linear1.bias.clone())
    
    def forward(self, x):
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
            
        # Use addmm for fused matrix multiplication and bias addition
        # This combines two operations into one CUDA kernel
        linear_output = torch.addmm(
            self.bias,    # bias
            x,            # input
            self.weight_t # transposed weight
        )
        
        # Apply sigmoid in-place to avoid allocation
        torch.sigmoid_(linear_output)
        
        # Sum along dimension 1 (hidden dimension)
        sum_output = torch.sum(linear_output, dim=1)
        
        # Optimized LogSumExp implementation
        max_val = torch.max(sum_output)
        
        # Subtract max_val for numerical stability (in-place)
        sum_output.sub_(max_val)
        
        # Compute exp in-place
        torch.exp_(sum_output)
        
        # Sum the exponentials
        sum_exp = torch.sum(sum_output)
        
        # Compute final result with log and adding back max_val
        result = torch.log(sum_exp) + max_val
        
        return result

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 128
input_size = 10
hidden_size = 20
output_size = 5

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [input_size, hidden_size, output_size]