import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs a series of operations: Gemm, Subtract, GlobalAvgPool, LogSumExp, GELU, and ResidualAdd.
    """
    def __init__(self, in_features, out_features, bias=True):
        super(Model, self).__init__()
        self.gemm = nn.Linear(in_features, out_features, bias=bias)
        self.subtract = nn.Parameter(torch.randn(out_features))
        
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        # On the first forward pass, capture the graph
        if self.graph is None:
            # Create static tensors for inputs. These tensors will be populated with
            # data from the real inputs on each forward pass.
            self.static_input = x.clone()

            # Create the graph object
            self.graph = torch.cuda.CUDAGraph()

            # Begin capturing the graph. All operations inside the 'with' block
            # will be recorded.
            with torch.cuda.graph(self.graph):
                # The entire original forward pass logic is replicated here,
                # but it operates on the static input tensor.
                original_x = self.static_input.clone().detach()
                # Gemm
                static_y = self.gemm(self.static_input)

                # Subtract
                static_y = static_y - self.subtract

                # GlobalAvgPool
                static_y = torch.mean(static_y, dim=1, keepdim=True)

                # LogSumExp
                static_y = torch.logsumexp(static_y, dim=1, keepdim=True)

                # GELU
                static_y = torch.nn.functional.gelu(static_y)

                # ResidualAdd
                static_y = static_y + original_x
            
            # The output of the captured graph is stored in a static tensor
            self.static_output = static_y

        # Copy the current input's data to the static input tensor
        self.static_input.copy_(x)

        # Replay the captured graph. This is much faster than executing
        # the operations in eager mode.
        self.graph.replay()

        # Return a clone of the static output tensor
        return self.static_output.clone()

batch_size = 128
in_features = 1024
out_features = 512

def get_inputs():
    return [torch.randn(batch_size, in_features)]

def get_init_inputs():
    return [in_features, out_features]