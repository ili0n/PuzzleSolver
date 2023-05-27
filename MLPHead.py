import torch.nn as nn
import torch.nn.functional as F
class MLPHead(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLPHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x