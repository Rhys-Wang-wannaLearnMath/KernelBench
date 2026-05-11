import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies sigmoid, and sums the result.
    """
    def __init__(self, input_size, hidden_size):
        super(Model, self).__init__()
        self.linear = nn.Linear(input_size, hidden_size)
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
        # A non-default stream is required for graph capture
        self.stream = torch.cuda.Stream()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, 1).
        """
        # Fallback to eager execution for non-CUDA tensors
        if not x.is_cuda:
            x = self.linear(x)
            x = torch.sigmoid(x)
            x = torch.sum(x, dim=1, keepdim=True)
            return x

        # On the first forward pass with a CUDA tensor, capture the graph.
        if self.graph is None:
            # Graph capture must be performed on a non-default stream.
            with torch.cuda.stream(self.stream):
                # Create a static input tensor that will be used for the graph.
                self.static_input = x.clone()

                # Instantiate the graph.
                self.graph = torch.cuda.CUDAGraph()

                # Begin capturing the graph.
                self.graph.capture_begin()

                # Run the model's operations to record them in the graph.
                y = self.linear(self.static_input)
                y = torch.sigmoid(y)
                self.static_output = torch.sum(y, dim=1, keepdim=True)

                # End capturing.
                self.graph.capture_end()

        # For all passes (including the first), copy the current input's data
        # to the static input tensor and replay the graph.
        self.static_input.copy_(x)
        self.graph.replay()

        # The result is now in the static output tensor.
        return self.static_output

batch_size = 128
input_size = 10
hidden_size = 20

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size]