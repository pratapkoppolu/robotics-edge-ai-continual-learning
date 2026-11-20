"""For smoke test ONLY: create disjoint current images by altering validation images."""
from pathlib import Path
from PIL import Image, ImageEnhance
from drift_detection.feature_extraction import image_paths
import argparse


def make(source, dest, factor=0.25):
    out = Path(dest)
    out.mkdir(parents=True, exist_ok=True)
    for p in image_paths(source):
        with Image.open(p) as img:
            image = ImageEnhance.Brightness(img.convert('RGB')).enhance(factor)
            image.save(out / p.name, quality=92)
    print(f'Generated {len(image_paths(source))} darkened images under {out}; smoke test only')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--brightness', type=float, default=0.25)
    a = parser.parse_args()
    make(a.source, a.out, a.brightness)
