import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, input_size, layer_sizes, output_size):
        """
        :param input_size: The number of input features
        :param layer_sizes: A list of ints containing the sizes of each hidden layer
        :param output_size: The number of output features
        """
        super(Model, self).__init__()
        
        layers = []
        current_input_size = input_size
        
        for layer_size in layer_sizes:
            layers.append(nn.Linear(current_input_size, layer_size))
            layers.append(nn.ReLU())
            current_input_size = layer_size
        
        layers.append(nn.Linear(current_input_size, output_size))
        
        self.network = nn.Sequential(*layers)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, input_size)
        :return: The output tensor, shape (batch_size, output_size)
        """
        if self.graph is None:
            # On the first forward pass, we perform a regular eager-mode execution
            # to get the result, and also capture the graph for future runs.
            
            # 1. Eager mode execution for the first input to ensure correctness.
            output = self.network(x)

            # 2. Capture the graph for subsequent executions.
            self.graph = torch.cuda.CUDAGraph()
            # Create static tensors that will be used as placeholders for graph replay.
            self.static_input = x.clone()
            
            with torch.cuda.graph(self.graph):
                self.static_output = self.network(self.static_input)

            # Return the result from the initial eager run.
            return output
        
        # For subsequent runs, use the captured graph.
        # 1. Copy the new input data to the memory location of the static input tensor.
        self.static_input.copy_(x)
        
        # 2. Replay the captured graph.
        self.graph.replay()
        
        # 3. Return a clone of the static output tensor.
        return self.static_output.clone()

# Test code
batch_size = 1
input_size = 1000
layer_sizes = [400, 800]
output_size = 500

def get_inputs():
    return [torch.randn(batch_size, input_size)]

def get_init_inputs():
    return [input_size, layer_sizes, output_size]