import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies GELU, and then applies Softmax.
    """
    def __init__(self, in_features, out_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None
        # A non-default stream is required for graph capture
        self.capture_stream = torch.cuda.Stream()

    def forward(self, x):
        # On the first run, capture the graph
        if self.graph is None:
            self.static_input = x
            
            # Graph capture must be done on a non-default stream.
            with torch.cuda.stream(self.capture_stream):
                self.graph = torch.cuda.CUDAGraph()
                self.graph.capture_begin()
                
                # Define the model's operations to be captured
                y = self.linear(self.static_input)
                y = torch.nn.functional.gelu(y)
                self.static_output = torch.nn.functional.softmax(y, dim=1)
                
                self.graph.capture_end()

            # Make the default stream wait for the capture to complete
            torch.cuda.current_stream().wait_stream(self.capture_stream)

        # For every run, copy the new input data and replay the graph.
        # This happens on the default stream.
        self.static_input.copy_(x)
        self.graph.replay()
        
        return self.static_output

batch_size = 128
in_features = 100
out_features = 10

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]