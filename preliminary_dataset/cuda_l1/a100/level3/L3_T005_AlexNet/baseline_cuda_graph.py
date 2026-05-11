import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_classes=1000):
        """
        :param num_classes: The number of output classes (default is 1000 for ImageNet)
        """
        super(Model, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Third convolutional layer
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        
        # Fourth convolutional layer
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        
        # Fifth convolutional layer
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(in_features=256 * 6 * 6, out_features=4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.0)
        
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.relu7 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.0)
        
        self.fc3 = nn.Linear(in_features=4096, out_features=num_classes)
        
        # CUDA Graph attributes
        self.graph = None
        self.static_input = None
        self.static_output = None
    
    def forward(self, x):
        """
        :param x: The input tensor, shape (batch_size, 3, 224, 224)
        :return: The output tensor, shape (batch_size, num_classes)
        """
        if self.graph is None:
            # On the first run, capture the graph
            self.graph = torch.cuda.CUDAGraph()
            self.static_input = x
            with torch.cuda.graph(self.graph):
                y = self.conv1(self.static_input)
                y = self.relu1(y)
                y = self.maxpool1(y)
                
                y = self.conv2(y)
                y = self.relu2(y)
                y = self.maxpool2(y)
                
                y = self.conv3(y)
                y = self.relu3(y)
                
                y = self.conv4(y)
                y = self.relu4(y)
                
                y = self.conv5(y)
                y = self.relu5(y)
                y = self.maxpool3(y)
                
                y = torch.flatten(y, 1)
                
                y = self.fc1(y)
                y = self.relu6(y)
                y = self.dropout1(y)
                
                y = self.fc2(y)
                y = self.relu7(y)
                y = self.dropout2(y)
                
                y = self.fc3(y)
                self.static_output = y

        # For all runs, copy the new input and replay the graph
        self.static_input.copy_(x)
        self.graph.replay()
        
        return self.static_output

# Test code
batch_size = 10
num_classes = 1000

def get_inputs():
    return [torch.randn(batch_size, 3, 224, 224)]

def get_init_inputs():
    return [num_classes]