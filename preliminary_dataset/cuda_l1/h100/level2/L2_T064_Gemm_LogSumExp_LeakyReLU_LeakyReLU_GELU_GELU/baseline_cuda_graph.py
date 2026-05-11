import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), followed by LogSumExp, LeakyReLU, 
    LeakyReLU, GELU, and GELU activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.graph is None:
            # On the first call, we need to initialize and capture the graph.
            # We create a static input tensor that will be used for all subsequent runs.
            # x.clone() is used to create a new tensor with its own memory.
            self.static_input = x.clone()

            # Create a new CUDA graph object.
            self.graph = torch.cuda.CUDAGraph()
            
            # Start capturing the graph. The operations inside this context manager
            # will be recorded. They are also run eagerly once to trace the graph.
            with torch.cuda.graph(self.graph):
                # Gemm
                y = self.linear(self.static_input)
                # LogSumExp
                y = torch.logsumexp(y, dim=1, keepdim=True)
                # LeakyReLU
                y = torch.nn.functional.leaky_relu(y, negative_slope=0.01)
                # LeakyReLU
                y = torch.nn.functional.leaky_relu(y, negative_slope=0.01)
                # GELU
                y = torch.nn.functional.gelu(y)
                # GELU
                y = torch.nn.functional.gelu(y)
                # The output of the traced operations is stored as the static output buffer.
                self.static_output = y

        # For every forward pass (including the first one after capture),
        # copy the data from the input tensor 'x' into our static input buffer.
        self.static_input.copy_(x)
        
        # Replay the captured graph. This executes the recorded operations
        # on the (updated) static_input and writes the result into static_output.
        self.graph.replay()
        
        # Return the tensor containing the result.
        return self.static_output

batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]