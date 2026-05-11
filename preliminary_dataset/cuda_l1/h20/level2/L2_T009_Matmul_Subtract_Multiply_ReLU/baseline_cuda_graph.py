import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, subtraction, multiplication, and ReLU activation.
    """
    def __init__(self, in_features, out_features, subtract_value, multiply_value):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.subtract_value = subtract_value
        self.multiply_value = multiply_value

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # Fallback to eager execution for non-CUDA tensors or during training
        if not x.is_cuda:
            x = self.linear(x)
            x = x - self.subtract_value
            x = x * self.multiply_value
            x = torch.relu(x)
            return x

        # First run: capture the graph
        if self.graph is None:
            self.graph = torch.cuda.CUDAGraph()
            # Use a clone of the input to create a static tensor
            self.static_input = x.clone()

            with torch.cuda.graph(self.graph):
                # The model's forward pass logic is captured here
                static_y = self.linear(self.static_input)
                static_y = static_y - self.subtract_value
                static_y = static_y * self.multiply_value
                self.static_output = torch.relu(static_y)

        # For all subsequent runs (and the first run after capture):
        # 1. Copy the new input data to the static input tensor
        self.static_input.copy_(x)

        # 2. Replay the graph
        self.graph.replay()

        # 3. Return a clone of the static output tensor
        return self.static_output.clone()

batch_size = 128
in_features = 10
out_features = 5
subtract_value = 2.0
multiply_value = 1.5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, subtract_value, multiply_value]