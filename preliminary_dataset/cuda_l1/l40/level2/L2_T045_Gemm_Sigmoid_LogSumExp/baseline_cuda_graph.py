import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication (Gemm), applies Sigmoid, sums the result, and calculates the LogSumExp.
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first run, capture the graph
        if self.graph is None:
            # Create static tensors. These will be used to capture the graph
            # and will be updated with new data on each forward pass.
            self.static_input = x.clone()

            # Instantiate the graph
            self.graph = torch.cuda.CUDAGraph()

            # Begin capturing on the default stream
            with torch.cuda.graph(self.graph):
                # The model's forward pass logic, using the static tensors
                static_y = self.linear1(self.static_input)
                static_y = torch.sigmoid(static_y)
                # The original model performs a two-step process here (sum then logsumexp).
                # We can keep this structure, but for CUDA graph purposes,
                # the intermediate result of sum() needs to be handled carefully.
                # However, the original code computes sum and then logsumexp on a different dimension,
                # so we will replicate that exact logic.
                static_y_sum = torch.sum(static_y, dim=1)
                self.static_output = torch.logsumexp(static_y_sum, dim=0)
        
        # Copy the current input's data to the static input tensor
        self.static_input.copy_(x)

        # Replay the graph. This executes the captured operations.
        self.graph.replay()

        # Return a clone of the static output tensor
        return self.static_output.clone()

batch_size = 128
input_size = 10
hidden_size = 20
output_size = 5

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, output_size]