import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Kullback-Leibler Divergence for comparing two distributions.

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
            self.static_predictions = torch.zeros_like(predictions)
            self.static_targets = torch.zeros_like(targets)
            
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.static_output = torch.nn.functional.kl_div(torch.log(self.static_predictions), self.static_targets, reduction='batchmean')

        self.static_predictions.copy_(predictions)
        self.static_targets.copy_(targets)
        
        self.graph.replay()
        
        return self.static_output

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape).softmax(dim=-1), torch.randn(batch_size, *input_shape).softmax(dim=-1)]

def get_init_inputs():
    return []