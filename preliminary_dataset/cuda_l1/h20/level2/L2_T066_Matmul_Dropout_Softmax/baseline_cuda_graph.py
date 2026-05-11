import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that performs matrix multiplication, applies dropout, calculates the mean, and then applies softmax.
    """
    def __init__(self, in_features, out_features, dropout_p):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout_p)
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
        # If the model is in training mode, dropout is stochastic and not
        # compatible with CUDA graphs. We run the model eagerly.
        # We also invalidate any existing graph.
        if self.training:
            self.graph = None
            x = self.matmul(x)
            x = self.dropout(x)
            x = torch.mean(x, dim=1, keepdim=True)
            x = torch.softmax(x, dim=1)
            return x

        # If in evaluation mode, we can use a CUDA graph.
        # If the graph has not been captured yet, we do it now.
        if self.graph is None:
            # Create a placeholder for the input tensor. Its data will be
            # replaced before each run.
            self.static_input = torch.empty_like(x)
            
            # Create the CUDA graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Begin capturing the graph.
            with torch.cuda.graph(self.graph):
                # Run the model's operations using the static placeholder input.
                # The operations are recorded into the graph.
                y = self.matmul(self.static_input)
                # In eval mode, dropout is a deterministic no-op.
                y = self.dropout(y)
                y = torch.mean(y, dim=1, keepdim=True)
                # The resulting output tensor is also static and part of the graph.
                self.static_output = torch.softmax(y, dim=1)

        # For every run (including the first one after capture), copy the
        # actual input data into the static input placeholder.
        self.static_input.copy_(x)

        # Replay the captured graph. This executes the recorded GPU operations
        # efficiently, updating the static_output tensor's memory with the new result.
        self.graph.replay()

        # Return the output tensor.
        return self.static_output

batch_size = 128
in_features = 100
out_features = 50
dropout_p = 0.2

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, dropout_p]