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
        # Initialize cudnn flags to their current global default values.
        # These can be modified on the model instance after creation.
        self.cudnn_enabled = torch.backends.cudnn.enabled
        self.cudnn_benchmark = torch.backends.cudnn.benchmark
        self.cudnn_deterministic = torch.backends.cudnn.deterministic
        self.cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32

        self.transition = nn.Sequential(
            nn.BatchNorm2d(num_input_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        """
        :param x: Input tensor of shape (batch_size, num_input_features, height, width)
        :return: Downsampled tensor with reduced number of feature maps
        """
        # Apply cudnn backend flags within the forward pass using a context manager
        with torch.backends.cudnn.flags(
            enabled=self.cudnn_enabled,
            benchmark=self.cudnn_benchmark,
            deterministic=self.cudnn_deterministic,
            allow_tf32=self.cudnn_allow_tf32,
        ):
            return self.transition(x)

batch_size = 10
num_input_features = 32
num_output_features = 64
height, width = 224, 224

def get_inputs():
    return [torch.randn(batch_size, num_input_features, height, width)]

def get_init_inputs():
    return [num_input_features, num_output_features]