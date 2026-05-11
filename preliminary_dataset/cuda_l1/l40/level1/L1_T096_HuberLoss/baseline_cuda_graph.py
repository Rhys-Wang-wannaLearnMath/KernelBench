import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Smooth L1 (Huber) Loss for regression tasks.

    Parameters:
        None
    """
    def __init__(self):
        super(Model, self).__init__()
        self.graph = None
        self.static_input_predictions = None
        self.static_input_targets = None
        self.static_output = None

    def forward(self, predictions, targets):
        if self.graph is None:
            # First run: capture the graph
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self.static_input_predictions = predictions.clone()
                self.static_input_targets = targets.clone()
                self.static_output = torch.nn.functional.smooth_l1_loss(self.static_input_predictions, self.static_input_targets)
            self.graph = g

        # For subsequent runs, copy the new input data and replay the graph
        self.static_input_predictions.copy_(predictions)
        self.static_input_targets.copy_(targets)
        self.graph.replay()
        return self.static_output

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []