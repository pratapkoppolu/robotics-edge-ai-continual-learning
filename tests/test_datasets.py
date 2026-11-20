from datasets.brushing_dataset import BrushingDataset
from datasets.picking_dataset import PickingDataset


def test_brushing_dataset():
    dataset = BrushingDataset(
        "datasets/task_a_brushing",
        split="train"
    )

    image, mask = dataset[0]

    assert image.shape == (3, 224, 224)
    assert mask.shape == (1, 224, 224)
    assert image.dtype.is_floating_point
    assert mask.dtype.is_floating_point


def test_picking_dataset():
    dataset = PickingDataset(
        "datasets/task_b_picking",
        split="train"
    )

    image, target = dataset[0]

    assert image.shape[0] == 3
    assert target["boxes"].shape[1] == 4
    assert len(target["boxes"]) == len(target["labels"])
