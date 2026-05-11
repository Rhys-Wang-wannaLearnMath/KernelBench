import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        """
        :param num_input_features: The number of input feature maps
        :param num_output_features: The number of output feature maps
        """
        super(Model, self).__init__()
        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        # Attributes for CUDA graph
        self.graph = None
        self.static_input = None
        self.static_output = None

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        if self.graph is None:
            # On the first run, execute the model eagerly to get the correct output
            # and update the state of any stateful modules (like BatchNorm).
            eager_output = self.transition(x)

            # After the first run, capture the graph.
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = x
            
            with torch.cuda.graph(self.graph):
                self.static_output = self.transition(self.static_input)
            
            return eager_output

        # For subsequent runs, replay the captured graph.
        # Copy the new input data into the static input tensor used for capture.
        self.static_input.copy_(x)
        self.graph.replay()
        # The result is now in the static output tensor.
        return self.static_output

batch_size = 10
num_input_features = 32
num_output_features = 64
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_input_features, num_output_features]