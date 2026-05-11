import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, applies Swish activation, sums with a bias term, and normalizes with GroupNorm.
    """
    def __init__(self, in_features, out_features, num_groups, bias_shape):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.bias = nn.Parameter(torch.randn(bias_shape))
        self.group_norm = nn.GroupNorm(num_groups, out_features)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        if self.graph is None:
            # On the first forward pass, we capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = torch.empty_like(x)

            with torch.cuda.graph(self.graph):
                # The model's forward logic is placed inside the capture block.
                # All operations are performed on the static tensors.
                static_y = self.matmul(self.static_input)
                static_y = torch.sigmoid(static_y) * static_y
                static_y = static_y + self.bias
                self.static_output = self.group_norm(static_y)

        # For all passes (including the first), we populate the static
        # input with the current input and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()

        # Return a clone of the static output tensor.
        return self.static_output.clone()

batch_size = 128
in_features = 512
out_features = 1024
num_groups = 32
bias_shape = (out_features,)

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, num_groups, bias_shape]