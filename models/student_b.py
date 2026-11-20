
import torch.nn as nn

from torchvision.models.detection import (
    fasterrcnn_mobilenet_v3_large_320_fpn
)

from models.shared_backbone import (
    get_foundation_weights
)


class StudentB(nn.Module):
    """
    Task B: Object detection for picking.

    Training:
        Returns a dictionary of detection losses.

    Inference:
        Returns boxes, labels and scores.
    """

    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        self.detector = (
            fasterrcnn_mobilenet_v3_large_320_fpn(
                weights=None,
                weights_backbone=get_foundation_weights(
                    pretrained
                ),
                num_classes=num_classes
            )
        )

    def forward(self, images, targets=None):
        return self.detector(images, targets)
