import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a value, applies Swish, Tanh, GELU, and Hardtanh activation functions.
    """
    def __init__(self, in_features, out_features, add_value_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.add_value = nn.Parameter(torch.randn(add_value_shape))
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first forward pass, the graph is not yet captured.
        if self.graph is None:
            # Create a new graph object.
            self.graph = torch.cuda.CUDAGraph()
            # Create a static input tensor with the same properties as the input.
            self.static_input = torch.empty_like(x)
            
            # Use the graph capture context manager.
            with torch.cuda.graph(self.graph):
                # Run the model's operations using the static input.
                # These operations are recorded into the graph.
                y = self.matmul(self.static_input)
                y = y + self.add_value
                y = torch.sigmoid(y) * y # Swish
                y = torch.tanh(y)
                y = torch.nn.functional.gelu(y) # GELU
                y = torch.nn.functional.hardtanh(y, min_val=-1, max_val=1) # Hardtanh
                # The final result is stored in a static output tensor.
                self.static_output = y

        # For every run (including the first), copy the current input data
        # into the static input tensor that the graph was created with.
        self.static_input.copy_(x)
        
        # Replay the captured graph. This executes the recorded operations
        # on the (now updated) static_input, writing to static_output.
        self.graph.replay()
        
        # Return the result from the static output tensor.
        return self.static_output

batch_size = 128
in_features = 1024
out_features = 512
add_value_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, add_value_shape]