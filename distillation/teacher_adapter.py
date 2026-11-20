"""Read only APPROVED task-compatible teacher annotations, never raw VLM prose.

Layout: <teacher_root>/<task>/<split>/<image_stem>.json. Each record must
include source_model, reviewer, approved:true, image_sha256, and task payload.
An independent human/QA process must create/approve these records.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for buf in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(buf)
    return h.hexdigest()


def load_record(teacher_root, task, split, image_path):
    image_path = Path(image_path)
    path = Path(teacher_root) / task / split / (image_path.stem + '.json')
    if not path.exists():
        raise FileNotFoundError(f'Approved teacher record missing: {path}')
    data = json.loads(path.read_text(encoding='utf-8'))
    for key in ('source_model', 'reviewer', 'approved', 'image_sha256'):
        if key not in data:
            raise ValueError(f'{path} missing {key}')
    if data['approved'] is not True or not str(data['reviewer']).strip():
        raise ValueError(f'{path}: teacher record is not approved')
    if sha256(image_path) != data['image_sha256']:
        raise ValueError(f'{path}: image hash mismatch')
    if data.get('task') != task:
        raise ValueError(f'{path}: task mismatch')
    return data


def segmentation_target(record, size):
    """Approved mask_path must be a real raster mask; no text-to-mask conversion."""
    mask_path = Path(record['mask_path'])
    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)
    mask = Image.open(mask_path).convert('L').resize(size, Image.Resampling.NEAREST)
    return torch.from_numpy((np.asarray(mask) > 0).astype('float32').copy()).unsqueeze(0)


def detection_target(record, image_size, num_classes):
    """Approved normalized YOLO boxes in 0-based object class IDs."""
    width, height = image_size
    boxes, labels = [], []
    for obj in record['objects']:
        cls = obj['class_id']
        xc, yc, bw, bh = obj['bbox_xywh_normalized']
        if not isinstance(cls, int) or not (0 <= cls < num_classes - 1):
            raise ValueError('Teacher class out of range')
        if not all(isinstance(v, (int, float)) and 0 <= v <= 1 for v in (xc, yc, bw, bh)):
            raise ValueError('Invalid normalized teacher bbox')
        x1, y1 = max(0., (xc-bw/2)*width), max(0., (yc-bh/2)*height)
        x2, y2 = min(float(width), (xc+bw/2)*width), min(float(height), (yc+bh/2)*height)
        if x2 <= x1 or y2 <= y1:
            raise ValueError('Degenerate teacher bbox')
        boxes.append([x1,y1,x2,y2]); labels.append(cls + 1)
    return {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1,4),
            'labels': torch.tensor(labels, dtype=torch.int64)}
