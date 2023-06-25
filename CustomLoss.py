import torch
import torch.nn as nn


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        batch_size, num_patches, num_classes = logits.size()
        loss = self.criterion(logits.view(batch_size * num_patches, num_classes), targets.view(-1))

        # Reshape logits and targets to group patches by images
        logits_grouped = logits.view(batch_size, num_patches, num_classes)
        targets_grouped = targets.view(batch_size, num_patches)

        # Apply penalty for repeated tags within each image
        for i in range(batch_size):
            unique_tags = torch.unique(targets_grouped[i])
            if unique_tags.size(0) < num_patches:
                repeated_tags = num_patches - unique_tags.size(0)
                loss += repeated_tags * 10  # Adjust penalty weight as needed

        return loss