"""Create synthetic domain-shift images for TRAIN and TEST, without touching source data.
Only synthetic experiment fixtures; never evidence of actual robot failures.
"""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps


def simulate_image(source, destination):
    with Image.open(source) as original:
        rgb = original.convert('RGB')
        # Large, deterministic lighting/colour shift; geometry unchanged so boxes remain valid.
        rgb = ImageEnhance.Brightness(rgb).enhance(0.42)
        rgb = ImageEnhance.Contrast(rgb).enhance(1.35)
        rgb = ImageOps.colorize(ImageOps.grayscale(rgb), black='#122341', white='#f0a65c')
        destination.parent.mkdir(parents=True, exist_ok=True)
        rgb.save(destination, quality=95)


def generate(source_root, destination_root, train_limit=8):
    src, dst = Path(source_root), Path(destination_root)
    if src.resolve() == dst.resolve():
        raise ValueError('Destination must not equal source')
    if dst.exists() and any(dst.rglob('*.jpg')):
        raise FileExistsError(f'Refusing to overwrite generated data: {dst}')
    records = {}
    for split in ('train', 'test'):
        files = sorted((src / split / 'images').glob('*.jpg'))
        if split == 'train':
            files = files[:train_limit]
        if not files:
            raise ValueError(f'No input images found for {split}')
        names = []
        for inp in files:
            label = src / split / 'labels' / (inp.stem + '.txt')
            if not label.is_file():
                raise FileNotFoundError(label)
            out_image = dst / split / 'images' / inp.name
            out_label = dst / split / 'labels' / label.name
            simulate_image(inp, out_image)
            out_label.parent.mkdir(parents=True, exist_ok=True)
            out_label.write_bytes(label.read_bytes())  # preserve geometry in synthetic shift
            names.append(inp.name)
        records[split] = names
    (dst / 'SIMULATION_ONLY.json').write_text(json.dumps({
        'simulation_only': True,
        'source': str(src),
        'transform': 'brightness_0.42_grayscale_colorized_contrast_1.35',
        'warning': 'Synthetic labels are inherited, not validated real observations',
        'files': records,
    }, indent=2), encoding='utf-8')
    return records


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default='datasets/task_b_picking')
    p.add_argument('--out', default='outputs/step11_simulated_b/new_data')
    p.add_argument('--train-limit', type=int, default=8)
    a = p.parse_args()
    print(json.dumps(generate(a.source, a.out, a.train_limit), indent=2))

if __name__ == '__main__':
    main()
