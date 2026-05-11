import torch
import torch.nn as nn

class Model(nn.Module):
    """
    A model that computes the Mean Squared Error loss for regression tasks.

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
        # If the graph has not been captured yet, record it on the first run.
        if self.graph is None:
            # First, perform an eager execution of the model to get the correct
            # output for the current inputs. This ensures correctness for the first pass.
            output = torch.mean((predictions - targets) ** 2)

            # Now, set up and capture the graph for all subsequent runs.
            # We clone the input tensors to create static tensors that are owned by the graph.
            # This prevents issues if the original input tensors are modified or deallocated.
            self.static_predictions = predictions.clone()
            self.static_targets = targets.clone()

            # Create the CUDA graph object.
            self.graph = torch.cuda.CUDAGraph()

            # Enter graph capture mode.
            with torch.cuda.graph(self.graph):
                # Define the graph's operations using the static tensors.
                # The output tensor is also created within the graph context, making it static.
                self.static_output = torch.mean((self.static_predictions - self.static_targets) ** 2)
            
            # Return the result from the initial eager run.
            return output
        else:
            # For all subsequent runs, the graph is already captured.
            # Update the data of the static input tensors with the new data.
            self.static_predictions.copy_(predictions)
            self.static_targets.copy_(targets)
            
            # Replay the captured graph. This executes the operations on the GPU
            # without the overhead of the Python interpreter. The result is written
            # into the static output tensor in-place.
            self.graph.replay()
            
            # Return the result.
            return self.static_output

batch_size = 128
input_shape = (4096, )
dim = 1

def get_inputs():
    return [torch.randn(batch_size, *input_shape), torch.randn(batch_size, *input_shape)]

def get_init_inputs():
    return []