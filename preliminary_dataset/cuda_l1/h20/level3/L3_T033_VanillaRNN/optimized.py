import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        """
        Initialize the Vanilla RNN model with optimized operations.
        
        :param input_size: The number of input features (int).
        :param hidden_size: The size of the hidden state (int).
        :param output_size: The number of output features (int).
        """
        super(ModelNew, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize hidden state just like the reference implementation
        self.hidden = torch.randn((batch_size, hidden_size))
        
        # Create temporary linear layers with the same initialization as the reference
        temp_i2h = nn.Linear(input_size + hidden_size, hidden_size)
        temp_h2o = nn.Linear(hidden_size, output_size)
        
        # Extract and separate the weights for input and hidden
        with torch.no_grad():
            # Split the i2h weights into input and hidden parts
            self.weight_ih = temp_i2h.weight[:, :input_size].clone()
            self.weight_hh = temp_i2h.weight[:, input_size:].clone()
            self.bias_h = temp_i2h.bias.clone()
            
            # Extract h2o weights
            self.weight_ho = temp_h2o.weight.clone()
            self.bias_o = temp_h2o.bias.clone()
        
        # Pre-transpose weights for faster matrix multiplication
        self.weight_ih_t = self.weight_ih.t().contiguous()
        self.weight_hh_t = self.weight_hh.t().contiguous()
        self.weight_ho_t = self.weight_ho.t().contiguous()
        
        # Flag to track if tensors have been moved to device
        self._device_initialized = False
        
        # Pre-allocate all buffers needed for computation
        self.hidden_buffer = None
        self.output_buffer = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the Vanilla RNN with optimized operations.
        
        :param x: Input tensor of shape (batch_size, input_size).
        :return: Output tensor of shape (batch_size, output_size).
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Move tensors to device only once or when device changes
        if not self._device_initialized or self.hidden.device != x.device:
            device = x.device
            self.hidden = self.hidden.to(device)
            self.weight_ih_t = self.weight_ih_t.to(device)
            self.weight_hh_t = self.weight_hh_t.to(device)
            self.weight_ho_t = self.weight_ho_t.to(device)
            self.bias_h = self.bias_h.to(device)
            self.bias_o = self.bias_o.to(device)
            
            # Initialize buffers on the correct device
            self.hidden_buffer = torch.empty((batch_size, self.hidden_size), device=device)
            self.output_buffer = torch.empty((batch_size, self.output_size), device=device)
            
            self._device_initialized = True
        
        # Compute hidden state using fused operations with pre-allocated buffer
        # First compute input contribution with bias
        torch.addmm(self.bias_h, x, self.weight_ih_t, out=self.hidden_buffer)
        
        # Add hidden contribution in-place
        self.hidden_buffer.addmm_(self.hidden, self.weight_hh_t)
        
        # Apply tanh activation in-place and update hidden state
        torch.tanh(self.hidden_buffer, out=self.hidden)
        
        # Compute output using fused operation with pre-allocated buffer
        torch.addmm(self.bias_o, self.hidden, self.weight_ho_t, out=self.output_buffer)
        
        return self.output_buffer

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 8
input_size = 1024
hidden_size = 256
output_size = 128
sequence_length = 256

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]