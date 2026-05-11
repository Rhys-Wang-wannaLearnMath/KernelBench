import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs a matrix multiplication, scaling, and residual addition.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        scaling_factor (float): Scaling factor to apply after matrix multiplication.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features).
        """
        # Fallback for CPU execution or if input is not a CUDA tensor
        if not x.is_cuda:
            y = self.matmul(x)
            original_y = y.clone().detach()
            y = y * self.scaling_factor
            y = y + original_y
            return y

        # On the first CUDA run, capture the graph.
        if self.graph is None:
            # Create a static input tensor that will be used as a placeholder.
            # Cloning ensures the graph's buffer is independent of the first input tensor.
            self.static_input = x.clone()

            # Create the graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Start capturing the graph on the default stream.
            with torch.cuda.graph(self.graph):
                # Run the model logic once to define the graph.
                # The resulting tensor becomes the static output placeholder for the graph.
                y_graph = self.matmul(self.static_input)
                original_y_graph = y_graph.clone().detach()
                y_graph = y_graph * self.scaling_factor
                self.static_output = y_graph + original_y_graph

        # For every CUDA run, copy the new data into the static input buffer and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()

        # Return a clone of the static output. This prevents the user from
        # accidentally modifying the graph's internal memory buffer.
        return self.static_output.clone()

batch_size = 128
in_features = 64
out_features = 128
scaling_factor = 0.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]