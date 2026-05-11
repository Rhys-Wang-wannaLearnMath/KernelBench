import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Cross Entropy Loss for multi-class classification tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()
        # Attributes for CUDA graph
        self.graph = None
        self.static_predictions = None
        self.static_targets = None
        self.static_output = None

    def forward(self, predictions, targets):
        # On the first forward pass, capture the graph
        if self.graph is None:
            # Create static placeholders for inputs
            self.static_predictions = torch.empty_like(predictions)
            self.static_targets = torch.empty_like(targets)

            # Create and capture the graph
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = torch.nn.functional.cross_entropy(self.static_predictions, self.static_targets)

        # For every forward pass (including the first), copy data and replay the graph
        self.static_predictions.copy_(predictions)
        self.static_targets.copy_(targets)
        self.graph.replay()

        # Return a clone of the output tensor to prevent modification of the graph's static tensor
        return self.static_output.clone()

batch_size = 4096
num_classes = 10
input_shape = (num_classes, )  # Output for each class
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randint(0, num_classes, (batch_size,))]

def get_init_inputs():
    return []