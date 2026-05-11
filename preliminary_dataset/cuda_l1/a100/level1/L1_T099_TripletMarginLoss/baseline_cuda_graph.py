import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes Triplet Margin Loss for metric learning tasks.

    Parameters:
        margin (float): The margin between the positive and negative samples.
    """
    def __init__(self, margin=1.0):
        super(Model, self).__init__()
        self.loss_fn = torch.nn.TripletMarginLoss(margin=margin)
        # Attributes for CUDA graph
        self.graph = None
        self.static_inputs = None
        self.static_output = None

    def forward(self, anchor, positive, negative):
        # On the first forward pass, we capture the graph.
        # We check for graph and also if the inputs are on a CUDA device.
        if self.graph is None and anchor.is_cuda:
            self.graph = torch.cuda.CUDAGraph()
            # Create static versions of the inputs. These tensors will have their
            # memory allocated and will be used to update the inputs for each run.
            self.static_inputs = [i.clone() for i in (anchor, positive, negative)]
            
            # Capture the graph
            with torch.cuda.graph(self.graph):
                self.static_output = self.loss_fn(*self.static_inputs)

        # If the graph has been captured, we can replay it.
        if self.graph is not None:
            # Copy the new input data into the static tensors
            for static_input, current_input in zip(self.static_inputs, (anchor, positive, negative)):
                static_input.copy_(current_input)
            
            # Replay the graph
            self.graph.replay()
            return self.static_output
        
        # Fallback for non-CUDA inputs or the very first (capturing) run
        return self.loss_fn(anchor, positive, negative)

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return [1.0]  # Default margin