
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

class BrushingDataset(Dataset):
    def __init__(self, root, split="train", size=(224, 224)):
        self.path = Path(root) / split
        self.images = sorted((self.path / "images").glob("*.jpg"))
        self.size = size

        if not self.images:
            raise ValueError(f"No images found: {self.path}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]
        mask_path = (
            self.path / "masks" / f"{image_path.stem}.png"
        )

        if not mask_path.exists():
            raise FileNotFoundError(mask_path)

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = image.resize(self.size, Image.Resampling.BILINEAR)
        mask = mask.resize(self.size, Image.Resampling.NEAREST)

        image = torch.from_numpy(
            np.asarray(image, dtype=np.float32)
            .transpose(2, 0, 1).copy() / 255.0
        )

        mask = torch.from_numpy(
            (np.asarray(mask, dtype=np.uint8) > 0)
            .astype(np.float32).copy()
        ).unsqueeze(0)

        return image, mask
