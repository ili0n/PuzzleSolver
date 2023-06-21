import torch.nn as nn
import torch.nn.functional as F


class CustomFFN(nn.Module):
    def __init__(self, hidden_size, num_patches):
        super(CustomFFN, self).__init__()
        self.hidden_size = hidden_size
        self.num_patches = num_patches

        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_patches),
        )
    def forward(self, hidden_states):
        # Flatten the hidden_states to (batch_size, num_patches, hidden_size)
        hidden_states = hidden_states[:, 1:, :]  # Exclude the CLS token
        hidden_states = hidden_states.view(-1, self.num_patches, self.hidden_size)


        # Apply the feed-forward network on each patch embedding
        logits = self.ffn(hidden_states)

        return logits
