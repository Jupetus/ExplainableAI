import torch.nn.functional as F
from torch import nn


class simpleCNN(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(simpleCNN, self).__init__()
        # Define the layers
        self.conv1 = nn.Conv2d(1, 3, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(3, 9, 5)
        self.fl1 = nn.Linear(input_size, hidden1)
        self.fl2 = nn.Linear(hidden1, hidden2)
        self.fl3 = nn.Linear(hidden2, output_size)
        self.out = nn.LogSoftmax(dim=1)
        
    # Forward pass
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 9 * 4 * 4)
        x = F.relu(self.fl1(x))
        x = F.relu(self.fl2(x))
        x = self.fl3(x)
        return self.out(x)
    