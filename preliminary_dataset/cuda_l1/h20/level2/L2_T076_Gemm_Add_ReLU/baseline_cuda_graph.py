import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, adds a bias term, and applies ReLU.
    """
    def __init__(self, in_features, out_features, bias_shape):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor with shape (batch_size, out_features).
        """
        # On the first forward pass, capture the model's operations in a CUDA graph.
        if self.graph is None:
            # Create a CUDA graph object.
            self.graph = torch.cuda.CUDAGraph()
            
            # Create a static tensor for capture. The graph will be captured on this tensor.
            self.static_input = x.clone()

            # Use the graph context manager to capture the operations.
            # By default, this uses a side stream, which is required for graph capture.
            with torch.cuda.graph(self.graph):
                y = self.gemm(self.static_input)
                y = y + self.bias
                self.static_output = torch.relu(y)
        
        # For all forward passes (including the first one after capture),
        # copy the new input data to the static input tensor's memory location.
        self.static_input.copy_(x)
        
        # Replay the captured graph.
        self.graph.replay()
        
        # Return the result from the static output tensor.
        return self.static_output

batch_size = 128
in_features = 1024
out_features = 512
bias_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, bias_shape]