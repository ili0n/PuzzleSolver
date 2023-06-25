import torch
import torch.nn as nn


class CustomLossNN(nn.Module):
    def __init__(self):
        super(CustomLossNN, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        batch_size, num_patches = logits.size()
        loss = self.criterion(logits,targets)

        # Reshape logits and targets to group patches by images
        logits_grouped = logits.view(batch_size, num_patches, -1)
        targets_grouped = targets.view(batch_size, -1)

        # Apply penalty for repeated tags within each image
        for i in range(batch_size):
            unique_tags = torch.unique(targets_grouped[i])
            if unique_tags.size(0) < num_patches:
                repeated_tags = num_patches - unique_tags.size(0)
                loss += repeated_tags * 10  # Adjust penalty weight as needed

        return loss