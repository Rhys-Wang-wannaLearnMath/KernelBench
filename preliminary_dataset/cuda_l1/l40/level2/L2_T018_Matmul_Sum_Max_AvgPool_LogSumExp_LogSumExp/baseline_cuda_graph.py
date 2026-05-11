import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a sequence of operations:
        - Matrix multiplication
        - Summation
        - Max
        - Average pooling
        - LogSumExp
        - LogSumExp
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        if self.graph is None:
            # On the first forward pass, we capture the graph.
            self.static_input = x
            
            # Create a new CUDA graph
            self.graph = torch.cuda.CUDAGraph()

            # Start capturing the graph
            with torch.cuda.graph(self.graph):
                # Place the model's operations inside the graph capture context
                # Use the static input tensor as the input to the captured operations
                graph_x = self.linear(self.static_input)  # (batch_size, out_features)
                graph_x = torch.sum(graph_x, dim=1, keepdim=True) # (batch_size, 1)
                graph_x = torch.max(graph_x, dim=1, keepdim=True)[0] # (batch_size, 1)
                graph_x = torch.mean(graph_x, dim=1, keepdim=True) # (batch_size, 1)
                graph_x = torch.logsumexp(graph_x, dim=1, keepdim=True) # (batch_size, 1)
                graph_x = torch.logsumexp(graph_x, dim=1, keepdim=True) # (batch_size, 1)
                
                # The final output of the captured operations is stored in a static output tensor
                self.static_output = graph_x

            # Replay the graph to perform the computation for the first input
            self.graph.replay()
            # Return a clone of the static output
            return self.static_output.clone()
        else:
            # For subsequent forward passes, the graph is already captured.
            # Copy the new input data to the static input tensor's memory
            self.static_input.copy_(x)
            # Replay the captured graph with the new input data
            self.graph.replay()
            # Return a clone of the static output
            return self.static_output.clone()

batch_size = 128
in_features = 10
out_features = 5

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]