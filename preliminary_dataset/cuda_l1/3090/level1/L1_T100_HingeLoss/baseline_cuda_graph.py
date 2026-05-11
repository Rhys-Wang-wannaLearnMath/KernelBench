import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Hinge Loss for binary classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()
        self.graph = None
        self.static_predictions = None
        self.static_targets = None
        self.static_output = None

    def forward(self, predictions, targets):
        if self.graph is None:
            # On the first forward pass, we capture the graph.
            # The input tensors from this first pass are used as placeholders.
            self.static_predictions = predictions
            self.static_targets = targets
            
            # Create a new CUDA graph
            self.graph = torch.cuda.CUDAGraph()
            
            # Start capturing operations into the graph
            with torch.cuda.graph(self.graph):
                # The operations are recorded, and the resulting tensor
                # is stored to be used as the output during replay.
                self.static_output = torch.mean(torch.clamp(1 - self.static_predictions * self.static_targets, min=0))

            # For the first call, we must execute the operations normally
            # to return the correct result, as capture doesn't execute.
            return torch.mean(torch.clamp(1 - predictions * targets, min=0))
        
        # For subsequent calls, we replay the captured graph.
        # First, copy the new input data into the placeholder tensors.
        self.static_predictions.copy_(predictions)
        self.static_targets.copy_(targets)
        
        # Replay the graph to compute the output.
        self.graph.replay()
        
        # Return the output tensor from the graph.
        return self.static_output

batch_size = 128
input_shape = (1,)
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, 2, (batch_size, 1)).float() * 2 - 1]

def get_init_inputs():
    return []