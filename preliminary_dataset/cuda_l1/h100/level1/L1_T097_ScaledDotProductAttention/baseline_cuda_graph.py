import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Cosine Similarity Loss for comparing vectors.

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
            # On the first forward pass, record the CUDA graph
            self.static_predictions = predictions.clone()
            self.static_targets = targets.clone()

            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = torch.mean(1 - torch.nn.functional.cosine_similarity(self.static_predictions, self.static_targets, dim=1))

        # For every run (including the first), copy the current input data to the static tensors
        self.static_predictions.copy_(predictions)
        self.static_targets.copy_(targets)

        # Replay the captured graph
        self.graph.replay()

        # Return a clone of the output tensor to avoid returning internal graph state
        return self.static_output.clone()

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []