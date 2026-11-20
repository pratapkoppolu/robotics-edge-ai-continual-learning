import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)


FOUNDATION_NAME = "MobileNetV3-Large"
FOUNDATION_VERSION = "foundation-v1"
FOUNDATION_WEIGHTS = MobileNet_V3_Large_Weights.IMAGENET1K_V1


def get_foundation_weights(pretrained=True):
    if pretrained:
        return FOUNDATION_WEIGHTS
    return None


def build_shared_backbone(pretrained=True):
    """
    Build an independent copy of the
    shared pretrained visual foundation.

    Returns:
        backbone: MobileNetV3-Large feature extractor
        output_channels: Number of output feature channels
    """

    model = mobilenet_v3_large(
        weights=get_foundation_weights(pretrained)
    )

    backbone = model.features

    output_channels = 960

    return backbone, output_channels
