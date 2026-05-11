import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a matrix multiplication, division, summation, and scaling.
    """
    def __init__(self, input_size, hidden_size, scaling_factor):
        super(Model, self).__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))
        self.scaling_factor = scaling_factor
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, hidden_size).
        """
        if self.graph is None:
            # On the first forward pass, run the model eagerly to get the correct output.
            # This ensures the first output is correct and determines the output tensor's shape.
            eager_output = torch.matmul(x, self.weight.T)
            eager_output = eager_output / 2
            eager_output = torch.sum(eager_output, dim=1, keepdim=True)
            eager_output = eager_output * self.scaling_factor

            # Initialize static tensors that will serve as persistent memory buffers for the graph.
            self.static_input = x.clone()
            self.static_output = eager_output.clone()

            # Now, capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                # The model's logic is placed inside the capture block, operating on the static input tensor.
                graphed_output = torch.matmul(self.static_input, self.weight.T)
                graphed_output = graphed_output / 2
                graphed_output = torch.sum(graphed_output, dim=1, keepdim=True)
                graphed_output = graphed_output * self.scaling_factor
                # The result of the graphed operations is copied into the static output buffer.
                self.static_output.copy_(graphed_output)

            # Return the result from the initial eager run.
            return eager_output
        
        # For all subsequent forward passes, use the captured graph.
        else:
            # Copy the new input data into the static input buffer.
            self.static_input.copy_(x)
            # Replay the graph. This executes the captured operations and writes the result into self.static_output.
            self.graph.replay()
            # Return a clone of the output from the static buffer.
            return self.static_output.clone()


batch_size = 128
input_size = 10
hidden_size = 20
scaling_factor = 1.5

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, hidden_size, scaling_factor]