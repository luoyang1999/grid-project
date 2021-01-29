import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # N = (W-F+2P)/S +1 输入尺寸W 卷积核F padding P 步长S
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = F.relu(self.conv1(x))   # input(3, 32, 32)  output(16, 28, 28)
        x = self.pool1(x)           # input(16, 28, 28) output(16, 14, 14)
        x = F.relu(self.conv2(x))   # input(16, 14, 14) output(32, 10, 10)
        x = self.pool2(x)           # input(32, 10, 10) output(32, 5, 5)
        x = x.view(-1, 32*5*5)      # input(32, 5, 5)   output(32*5*5)
        x = F.relu(self.fc1(x))     # input(32*5*5)     output(120)
        x = F.relu(self.fc2(x))     # input(120)        output(84)
        x = self.fc3(x)             # input(84)         output(10)
        return x

import torch
input1 = torch.rand([32, 3, 32, 32])
model = LeNet()
print(model)
output = model(input1)
