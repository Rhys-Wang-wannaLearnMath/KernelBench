import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Performs batched matrix multiplication (C = A * B) where A, B, and C have the same batch dimension.
    """
    def __init__(self):
        super(Model, self).__init__()
        # Attributes for CUDA graph
        self.graph = None
        self.static_A = None
        self.static_B = None
        self.static_C = None
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Performs batched matrix multiplication.

        Args:
            A: Input tensor of shape (batch_size, m, k).
            B: Input tensor of shape (batch_size, k, n).

        Returns:
            C: Output tensor of shape (batch_size, m, n).
        """
        # On the first forward pass, capture the graph
        if self.graph is None:
            # Create static tensors with the same properties as the inputs
            # These will be the placeholders for the graph
            self.static_A = torch.zeros_like(A)
            self.static_B = torch.zeros_like(B)
            self.static_C = torch.zeros(
                A.shape[0], A.shape[1], B.shape[2], dtype=A.dtype, device=A.device
            )

            # Create the graph object
            self.graph = torch.cuda.CUDAGraph()
            
            # Begin graph capture
            with torch.cuda.graph(self.graph):
                # The operation to be captured.
                # The result is copied into the static output tensor.
                graphed_C = torch.bmm(self.static_A, self.static_B)
                self.static_C.copy_(graphed_C)
        
        # Copy the current input data to the static tensors
        self.static_A.copy_(A)
        self.static_B.copy_(B)
        
        # Replay the graph. The result will be in self.static_C
        self.graph.replay()
        
        # Return a clone of the static output tensor
        return self.static_C.clone()

batch_size = 128
m = 128
k = 256
n = 512

def get_inputs():
    A = torch.randn(batch_size, m, k)
    B = torch.randn(batch_size, k, n)
    return [A, B]

def get_init_inputs():
    return []  # No special initialization inputs needed