import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.0):
        """
        Initialize the LSTM model with optimized CUDA implementation.

        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param num_layers: Number of recurrent layers
        :param output_size: The number of output features
        :param dropout: If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to `dropout`
        """
        super(ModelNew, self).__init__()
        
        # Register hidden states as buffers for automatic device management
        self.register_buffer('h0', torch.randn((num_layers, batch_size, hidden_size)))
        self.register_buffer('c0', torch.randn((num_layers, batch_size, hidden_size)))
        
        # Create the LSTM layer with optimal configuration
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )
        
        # Linear layer for output
        self.fc = nn.Linear(hidden_size, output_size)
        
        # Enable cuDNN benchmarking for optimal algorithm selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
        
        # CUDA graph optimization
        self.graph_cache = {}  # Cache for multiple input shapes
        self.use_cuda_graph = torch.cuda.is_available()
        self.warmup_count = 3  # Optimal warmup count based on previous experiments
        
    def _warmup(self, x):
        """Perform warmup iterations to stabilize performance"""
        with torch.no_grad():
            for _ in range(self.warmup_count):
                out, _ = self.lstm(x, (self.h0, self.c0))
                self.fc(out[:, -1, :])
    
    def _create_cuda_graph(self, x):
        """Create and capture CUDA graph for the forward pass"""
        try:
            # Create static inputs for CUDA graph capture
            static_input = torch.zeros_like(x, device=x.device)
            graph_output = torch.zeros((x.size(0), self.fc.out_features), device=x.device)
            
            # Perform warmup iterations
            self._warmup(x)
            
            # Capture the graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                # Forward pass through LSTM
                out, _ = self.lstm(static_input, (self.h0, self.c0))
                # Extract last timestep and pass through linear layer
                out = self.fc(out[:, -1, :])
                graph_output.copy_(out)
            
            # Cache the graph and associated tensors
            shape_key = (x.shape[0], x.shape[1], x.shape[2])
            self.graph_cache[shape_key] = {
                'graph': g,
                'static_input': static_input,
                'output': graph_output
            }
            
            # Run the graph once to ensure everything is initialized
            static_input.copy_(x, non_blocking=True)
            g.replay()
            return True
        except Exception:
            # Fallback to standard execution if graph capture fails
            return False
    
    def forward(self, x):
        """
        Forward pass through the LSTM model.

        :param x: The input tensor, shape (batch_size, sequence_length, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Fast path: ensure tensors are on the correct device
        device = self.h0.device
        if x.device != device:
            x = x.to(device, non_blocking=True)
        
        # Ensure input is contiguous for better memory access patterns
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Fast path: Use CUDA graph if available
        if self.use_cuda_graph and x.is_cuda:
            shape_key = (x.shape[0], x.shape[1], x.shape[2])
            
            # Check if we have a cached graph for this input shape
            if shape_key in self.graph_cache:
                cached = self.graph_cache[shape_key]
                cached['static_input'].copy_(x, non_blocking=True)
                cached['graph'].replay()
                return cached['output']
            
            # Create new graph for this input shape
            if self._create_cuda_graph(x):
                cached = self.graph_cache[shape_key]
                return cached['output']
        
        # Fallback path: Standard forward pass if CUDA graph is not used
        out, _ = self.lstm(x, (self.h0, self.c0))
        out = self.fc(out[:, -1, :])
        
        return out

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
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