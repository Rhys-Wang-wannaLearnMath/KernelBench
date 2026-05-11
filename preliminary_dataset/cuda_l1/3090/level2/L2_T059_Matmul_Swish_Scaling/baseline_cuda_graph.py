import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a matrix multiplication, applies Swish activation, and scales the result.
    """
    def __init__(self, in_features, out_features, scaling_factor):
        super(Model, self).__init__()
        self.matmul = nn.Linear(in_features, out_features)
        self.scaling_factor = scaling_factor
        
        # Attributes for CUDA graph
        self.cuda_graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        if self.cuda_graph is None:
            # On the first pass, run the model to get a real output.
            # This serves as the result for the first call and is used to allocate a static output buffer.
            real_output = self.matmul(x)
            real_output = real_output * torch.sigmoid(real_output)  # Swish activation
            real_output = real_output * self.scaling_factor

            # Pre-allocate static buffers for graph inputs and outputs.
            self.static_input = x.clone()
            self.static_output = real_output.clone()
            
            # Capture the graph.
            self.cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.cuda_graph):
                # The graphed operations use the static buffers.
                # The final operation writes its result into the pre-allocated static_output buffer.
                graphed_y = self.matmul(self.static_input)
                graphed_y = graphed_y * torch.sigmoid(graphed_y)
                torch.multiply(graphed_y, self.scaling_factor, out=self.static_output)
            
            return real_output
        else:
            # For subsequent passes, copy the new input data into the static buffer and replay the graph.
            self.static_input.copy_(x)
            self.cuda_graph.replay()
            # The result is now in the static_output buffer.
            return self.static_output.clone()

batch_size = 128
in_features = 1024
out_features = 512
scaling_factor = 2.0

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features, scaling_factor]