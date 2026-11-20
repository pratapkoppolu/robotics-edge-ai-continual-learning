
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.shared_backbone import build_shared_backbone


class StudentA(nn.Module):
    """
    Task A: Brushing region segmentation.

    Input:
        [B, 3, H, W]

    Output:
        [B, 1, H, W] segmentation logits
    """

    def __init__(self, pretrained=True):
        super().__init__()

        self.backbone, channels = build_shared_backbone(
            pretrained=pretrained
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(
                channels, 128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128, 64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1)
        )

    def forward(self, images):
        input_size = images.shape[-2:]

        features = self.backbone(images)
        logits = self.decoder(features)

        logits = F.interpolate(
            logits,
            size=input_size,
            mode="bilinear",
            align_corners=False
        )

        return logits
