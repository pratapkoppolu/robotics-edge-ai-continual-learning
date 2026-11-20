
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
random.seed(42)

SPLITS = {
    "train": 40,
    "val": 10,
    "test": 10
}

IMAGE_SIZE = 224


def generate_task_a():
    """Create brushing images and binary masks."""

    for split, count in SPLITS.items():
        base = ROOT / "task_a_brushing" / split
        images_dir = base / "images"
        masks_dir = base / "masks"

        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            image = Image.new(
                "RGB", (IMAGE_SIZE, IMAGE_SIZE),
                color=(180, 180, 180)
            )

            mask = Image.new(
                "L", (IMAGE_SIZE, IMAGE_SIZE), 0
            )

            draw_image = ImageDraw.Draw(image)
            draw_mask = ImageDraw.Draw(mask)

            x1 = random.randint(20, 90)
            y1 = random.randint(20, 90)
            x2 = random.randint(130, 200)
            y2 = random.randint(130, 200)

            box = (x1, y1, x2, y2)

            draw_image.ellipse(box, fill=(50, 120, 210))
            draw_mask.ellipse(box, fill=255)

            filename = f"sample_{i:04d}"

            image.save(images_dir / f"{filename}.jpg")
            mask.save(masks_dir / f"{filename}.png")


def generate_task_b():
    """Create picking images and YOLO-format labels."""

    for split, count in SPLITS.items():
        base = ROOT / "task_b_picking" / split
        images_dir = base / "images"
        labels_dir = base / "labels"

        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for i in range(count):
            image = Image.new(
                "RGB", (IMAGE_SIZE, IMAGE_SIZE),
                color=(200, 200, 200)
            )

            draw = ImageDraw.Draw(image)

            class_id = random.randint(0, 2)

            x1 = random.randint(20, 90)
            y1 = random.randint(20, 90)
            x2 = random.randint(130, 200)
            y2 = random.randint(130, 200)

            box = (x1, y1, x2, y2)

            if class_id == 0:
                draw.rectangle(box, fill=(220, 60, 60))
            elif class_id == 1:
                draw.ellipse(box, fill=(60, 180, 60))
            else:
                draw.rectangle(box, fill=(60, 60, 220))

            xc = ((x1 + x2) / 2) / IMAGE_SIZE
            yc = ((y1 + y2) / 2) / IMAGE_SIZE
            w = (x2 - x1) / IMAGE_SIZE
            h = (y2 - y1) / IMAGE_SIZE

            filename = f"sample_{i:04d}"

            image.save(images_dir / f"{filename}.jpg")

            label = (
                f"{class_id} "
                f"{xc:.6f} {yc:.6f} "
                f"{w:.6f} {h:.6f}\n"
            )

            (labels_dir / f"{filename}.txt").write_text(
                label
            )


if __name__ == "__main__":
    generate_task_a()
    generate_task_b()

    print("Synthetic datasets generated successfully.")
