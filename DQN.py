import torch
import torch.nn as nn
import torch.nn.functional as F

class DNQ(nn.Module):
    def __init__(self, output_size=9):
        super(DNQ, self).__init__()
        #conv 4,84,84-> 16, 20,20
        self.conv1 = nn.Conv2d(4,16,kernel_size=8,stride=4)
        #conv 16,20,20-> 32,9,9

        self.conv2 = nn.Conv2d(16,32,kernel_size=4,stride=2)

        self.fc1 = nn.Linear(32 * 9 * 9 ,256)
        self.fc2 = nn.Linear(256 ,output_size)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        #flatten to fit the fc
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        
        return (self.fc2(x))