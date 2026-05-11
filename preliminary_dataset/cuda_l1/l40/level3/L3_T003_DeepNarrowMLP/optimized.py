import torch
import torch.nn as nn
import math

class ModelNew(nn.Module):
    def __init__(self, input_size, hidden_layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param hidden_layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(ModelNew, self).__init__()
        
        # Pre-allocate weights and biases in contiguous memory
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        
        layer_sizes = [input_size] + hidden_layer_sizes + [output_size]
        
        # Initialize all layers at once to improve memory locality
        for i in range(1, len(layer_sizes)):
            out_size = layer_sizes[i]
            in_size = layer_sizes[i-1]
            
            # Initialize weight with same method as nn.Linear
            weight = torch.empty(out_size, in_size)
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            
            # Store pre-transposed for efficient matrix multiplication
            self.weights.append(nn.Parameter(weight.t().contiguous()))
            
            # Initialize bias with same method as nn.Linear
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
            bound = 1 / math.sqrt(fan_in)
            bias = torch.empty(out_size)
            nn.init.uniform_(bias, -bound, bound)
            self.biases.append(nn.Parameter(bias))
        
        # Cache direct references to parameters for faster access
        self.cached_weights = [w for w in self.weights]
        self.cached_biases = [b for b in self.biases]
        
        # Number of hidden layers (excluding output layer)
        self.num_hidden = len(hidden_layer_sizes)
        
        # Enable cudnn benchmarking for optimized kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        # Ensure input is contiguous for optimal memory access
        if not x.is_contiguous():
            x = x.contiguous()
        
        # Process hidden layers with optimized operations
        for i in range(self.num_hidden):
            # Efficient matrix multiplication with bias addition
            x = torch.addmm(self.cached_biases[i], x, self.cached_weights[i])
            # In-place ReLU activation
            x.relu_()
        
        # Output layer (no activation)
        x = torch.addmm(self.cached_biases[-1], x, self.cached_weights[-1])
        
        return x

# CRITICAL: Keep ALL hyperparameters EXACTLY as shown in the reference implementation
batch_size = 1
input_size = 1000
hidden_layer_sizes = [50, 50, 50, 50, 50, 50, 50, 50]  # Example of deep and narrow layers
output_size = 10

def get_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    # Use the EXACT same hyperparameters as in the reference implementation
    return [input_size, hidden_layer_sizes, output_size]