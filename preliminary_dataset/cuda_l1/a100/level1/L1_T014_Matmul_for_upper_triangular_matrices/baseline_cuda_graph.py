import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs matrix multiplication (C = A * B) for upper triangular matrices.
    """
    def __init__(self):
        super(Model, self).__init__()
        self.graph = None
        self.static_inputs = None
        self.static_output = None
    
    def forward(self, A, B):
        """
        Performs matrix multiplication for upper triangular matrices.

        Args:
            A (torch.Tensor): Upper triangular matrix of shape (N, N).
            B (torch.Tensor): Upper triangular matrix of shape (N, N).

        Returns:
            torch.Tensor: The product of A and B, also an upper triangular matrix of shape (N, N).
        """
        if self.graph is None:
            # On the first run, capture the CUDA graph
            self.static_inputs = [A.clone(), B.clone()]
            
            # Create a static output tensor. Its shape is determined by a
            # "meta" run of the operations, which is fast.
            self.static_output = torch.empty_like(torch.triu(torch.matmul(A, B)))

            # Create the graph object
            self.graph = torch.cuda.CUDAGraph()
            
            # Begin capturing the graph
            with torch.cuda.graph(self.graph):
                # Run the model's operations using the static inputs. These operations are recorded.
                graphed_output = torch.triu(torch.matmul(self.static_inputs[0], self.static_inputs[1]))
                # The result of the captured operations is copied to the static output tensor.
                self.static_output.copy_(graphed_output)

        # For every run (including the first), copy the current input data to the static tensors
        self.static_inputs[0].copy_(A)
        self.static_inputs[1].copy_(B)

        # Replay the captured graph. This executes the recorded operations with the new input data.
        self.graph.replay()

        # Return the static output tensor, which now holds the latest result.
        return self.static_output

N = 4096

def get_inputs():
    """
    Generates upper triangular matrices for testing.

    Returns:
        list: A list containing two upper triangular matrices of shape (N, N).
    """
    A = torch.triu(torch.randn(N, N))
    B = torch.triu(torch.randn(N, N))
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs are needed for this model.

    Returns:
        list: An empty list.
    """
    return []