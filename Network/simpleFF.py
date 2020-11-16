import torch.nn.functional as F
from torch import nn

class simpleFF(nn.Module):
    
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(simpleFF, self).__init__()
        # Define the layers
        self.fl1 = nn.Linear(input_size, hidden1)
        self.fl2 = nn.Linear(hidden1, hidden2)
        self.fl3 = nn.Linear(hidden2, output_size)
        self.out = nn.LogSoftmax(dim=1)
        
    # Forward pass
    def forward(self, x):
        x = F.relu(self.fl1(x))
        x = F.relu(self.fl2(x))
        x = self.fl3(x)
        return self.out(x)
        