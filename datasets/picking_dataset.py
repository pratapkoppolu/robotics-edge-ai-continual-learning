
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class PickingDataset(Dataset):
    def __init__(self, root, split="train"):
        self.path = Path(root) / split
        self.images = sorted((self.path / "images").glob("*.jpg"))

        if not self.images:
            raise ValueError(f"No images found: {self.path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        label_path = (
            self.path / "labels" / f"{image_path.stem}.txt"
        )

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        boxes, labels = [], []

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue

            cls, xc, yc, bw, bh = map(float, line.split())

            if not all(0 <= v <= 1 for v in [xc, yc, bw, bh]):
                raise ValueError(f"Invalid box: {label_path}")

            x1 = max(0, (xc - bw / 2) * width)
            y1 = max(0, (yc - bh / 2) * height)
            x2 = min(width, (xc + bw / 2) * width)
            y2 = min(height, (yc + bh / 2) * height)

            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Degenerate box: {label_path}")

            boxes.append([x1, y1, x2, y2])
            labels.append(int(cls) + 1)

        image_tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32)
            .transpose(2, 0, 1).copy() / 255.0
        )

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor(index)
        }

        return image_tensor, target
