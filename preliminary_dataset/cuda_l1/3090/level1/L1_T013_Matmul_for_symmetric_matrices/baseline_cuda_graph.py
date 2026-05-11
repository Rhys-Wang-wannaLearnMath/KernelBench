import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Simple model that performs a single matrix multiplication (C = A * B) with A and B being symmetric matrices.
    """
    def __init__(self):
        super(Model, self).__init__()
        # State for CUDA graph. A private, non-default stream is required for capture.
        self.stream = torch.cuda.Stream()
        self.graph = None
        self.static_inputs = None
        self.static_output = None
    
    def forward(self, A, B):
        """
        Performs matrix multiplication of two symmetric matrices.

        Args:
            A (torch.Tensor): Input matrix A, shape (N, N), symmetric.
            B (torch.Tensor): Input matrix B, shape (N, N), symmetric.

        Returns:
            torch.Tensor: Output matrix C, shape (N, N).
        """
        # On the first forward pass, the graph is not yet captured.
        if self.graph is None:
            # First, perform a regular eager-mode execution to get the correct output for this call.
            # This ensures the first run's result is correct and matches the original model's behavior.
            C_eager = torch.matmul(A, B)
            
            # Then, capture the graph on the private stream for all future invocations.
            with torch.cuda.stream(self.stream):
                # Use clones of the inputs as static placeholders for the graph.
                self.static_inputs = [A.clone(), B.clone()]
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.static_output = torch.matmul(self.static_inputs[0], self.static_inputs[1])

            # Synchronize the private stream to ensure graph capture is complete before the next forward call.
            self.stream.synchronize()

            # Return the eagerly computed result for this first run.
            return C_eager

        # For all subsequent calls, the graph has been captured.
        # Update the static inputs with the new data.
        self.static_inputs[0].copy_(A)
        self.static_inputs[1].copy_(B)
        
        # Replay the captured graph. The operations are executed on the current stream.
        self.graph.replay()
        
        # Return the result tensor, which has been updated by the graph replay.
        return self.static_output

N = 4096

def get_inputs():
    """
    Generates a pair of random symmetric matrices for testing.

    Returns:
        list: List containing two symmetric tensors A and B.
    """
    A = torch.randn(N, N)
    A = (A + A.T) / 2  # Ensure symmetry
    B = torch.randn(N, N)
    B = (B + B.T) / 2  # Ensure symmetry
    return [A, B]

def get_init_inputs():
    """
    No specific initialization inputs needed for this model.

    Returns:
        list: Empty list.
    """
    return []