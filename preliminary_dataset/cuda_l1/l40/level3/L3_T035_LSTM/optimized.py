import torch
import torch.nn as nn

class LastTimeStepLinearFunction(torch.autograd.Function):
    """
    Custom autograd function that fuses the extraction of the last time step
    from LSTM output and the linear transformation.
    """
    @staticmethod
    def forward(ctx, lstm_output, weight, bias):
        # Extract the last time step from each sequence
        last_seq = lstm_output[:, -1, :]
        
        # Apply linear transformation
        output = torch.matmul(last_seq, weight.t())
        if bias is not None:
            output += bias
        
        # Save for backward
        ctx.save_for_backward(last_seq, weight, bias)
        ctx.lstm_output_shape = lstm_output.shape
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        last_seq, weight, bias = ctx.saved_tensors
        lstm_output_shape = ctx.lstm_output_shape
        
        # Gradient for weight
        grad_weight = torch.matmul(grad_output.t(), last_seq)
        
        # Gradient for bias
        grad_bias = None
        if bias is not None:
            grad_bias = grad_output.sum(0)
        
        # Gradient for lstm_output
        # Only the last time step gets gradient
        grad_last_seq = torch.matmul(grad_output, weight)
        grad_lstm_output = torch.zeros(lstm_output_shape, device=grad_output.device, dtype=grad_output.dtype)
        grad_lstm_output[:, -1, :] = grad_last_seq
        
        return grad_lstm_output, grad_weight, grad_bias

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model with optimized operations.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer
        """
        super(ModelNew, self).__init__()
        # Initialize hidden state with random values - exactly as in reference implementation
        self.h0 = torch.randn((num_layers, batch_size, hidden_size))
        self.c0 = torch.randn((num_layers, batch_size, hidden_size))
        
        # Use PyTorch's optimized LSTM implementation
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, 
                           dropout=dropout, bidirectional=False)
        
        # Linear layer for final output
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Custom function for fused operations
        self.fused_op = LastTimeStepLinearFunction.apply
        
        # CUDA graph related
        self.graph = None
        self.static_input = None
        self.static_h0 = None
        self.static_c0 = None
        self.static_output = None
        self.warmup_done = False
        
        # Device tracking for hidden states
        self.device_h0 = None
        self.device_c0 = None
        self.last_device = None
        self.last_input_shape = None
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Optimize hidden state handling - avoid repeated transfers
        if self.device_h0 is None or self.last_device != x.device:
            self.device_h0 = self.h0.to(x.device)
            self.device_c0 = self.c0.to(x.device)
            self.last_device = x.device
            # Reset graph state when device changes
            self.graph = None
            self.warmup_done = False
        
        # Use CUDA graphs for repeated executions with same input shape on CUDA devices
        if x.is_cuda and torch.cuda.is_available():
            try:
                # Check if input shape has changed
                if self.last_input_shape != x.shape:
                    self.last_input_shape = x.shape
                    self.graph = None
                    self.warmup_done = False
                    self.static_input = None
                
                # Initialize static tensors if needed
                if self.static_input is None:
                    self.static_input = torch.zeros_like(x)
                    self.static_h0 = torch.zeros_like(self.device_h0)
                    self.static_c0 = torch.zeros_like(self.device_c0)
                    self.static_output = torch.zeros((x.size(0), self.fc.out_features), device=x.device)
                
                # Copy input data to static tensors
                self.static_input.copy_(x)
                self.static_h0.copy_(self.device_h0)
                self.static_c0.copy_(self.device_c0)
                
                if not self.warmup_done:
                    # Run once without graph to warm up
                    lstm_output, _ = self.lstm(self.static_input, (self.static_h0, self.static_c0))
                    output = self.fused_op(lstm_output, self.fc.weight, self.fc.bias)
                    self.static_output.copy_(output)
                    torch.cuda.synchronize()
                    
                    # Now capture the graph
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g):
                        lstm_output, _ = self.lstm(self.static_input, (self.static_h0, self.static_c0))
                        output = self.fused_op(lstm_output, self.fc.weight, self.fc.bias)
                        self.static_output.copy_(output)
                    self.graph = g
                    self.warmup_done = True
                
                # Replay the graph
                self.graph.replay()
                return self.static_output
            
            except Exception:
                # Fallback in case of CUDA graph error
                self.warmup_done = False
                self.graph = None
        
        # Standard execution path for non-CUDA or when CUDA graph fails
        lstm_output, _ = self.lstm(x, (self.device_h0, self.device_c0))
        return self.fused_op(lstm_output, self.fc.weight, self.fc.bias)

# Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 10
sequence_length = 512
input_size = 128
hidden_size = 256
num_layers = 6
output_size = 10
dropout = 0.0

def get_inputs():
    return [torch.randn(batch_size, sequence_length, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, num_layers, output_size, dropout]