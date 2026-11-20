
import torch
import torch.nn as nn


class BrushingLoss(nn.Module):
    def __init__(self, dice_weight=0.5, eps=1e-6):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)

        intersection = (
            probs * targets
        ).sum(dim=(1, 2, 3))

        denominator = (
            probs.sum(dim=(1, 2, 3))
            + targets.sum(dim=(1, 2, 3))
        )

        dice_loss = 1 - (
            (2 * intersection + self.eps)
            / (denominator + self.eps)
        ).mean()

        return (
            (1 - self.dice_weight) * bce_loss
            + self.dice_weight * dice_loss
        )
