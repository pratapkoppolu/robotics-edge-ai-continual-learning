"""M7: Frozen, version-pinned MobileNetV3 visual embeddings."""
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

WEIGHTS = MobileNet_V3_Large_Weights.IMAGENET1K_V1
ENCODER_ID = 'mobilenet_v3_large_IMAGENET1K_V1_features_avgpool_v1'


def image_paths(folder):
    folder = Path(folder)
    paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}) if folder.is_dir() else []
    if not paths:
        raise ValueError(f'No images in {folder}')
    return paths


def extract(folder, output, batch_size=8, device='cpu'):
    """Extract and save frozen 960-d global-pooled features, filenames, hashes, encoder ID."""
    paths = image_paths(folder)
    model = mobilenet_v3_large(weights=WEIGHTS).features.eval().to(device)
    for param in model.parameters():
        param.requires_grad_(False)
    transform = WEIGHTS.transforms()
    vectors = []
    with torch.inference_mode():
        for start in range(0, len(paths), batch_size):
            chunk = paths[start:start + batch_size]
            batch = torch.stack([transform(Image.open(p).convert('RGB')) for p in chunk]).to(device)
            features = model(batch).mean(dim=(2, 3))
            vectors.append(features.cpu().numpy().astype(np.float32))
    matrix = np.concatenate(vectors, axis=0)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
    np.savez_compressed(output, embeddings=matrix, filenames=np.asarray([str(p) for p in paths]), hashes=np.asarray(hashes), encoder_id=np.asarray(ENCODER_ID))
    return {'output': str(output), 'images': len(paths), 'shape': list(matrix.shape), 'encoder_id': ENCODER_ID}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    print(json.dumps(extract(args.images, args.out, args.batch_size, args.device), indent=2))
