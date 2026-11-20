"""Create leakage-safe E0/E1 experiment manifests; never alter source data.
Run: python -m distillation.prepare_e0_e1 --dataset-root datasets/task_b_picking
"""
import argparse
import hashlib
import json
import random
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def build(dataset_root, approved_root, out_dir, seed=42, labeled_count=20):
    dataset_root, approved_root, out_dir = map(Path, (dataset_root, approved_root, out_dir))
    train_dir = dataset_root / 'train'
    images = sorted(p for p in (train_dir / 'images').iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    if not images:
        raise ValueError(f'No images found: {train_dir / "images"}')
    if not (1 <= labeled_count < len(images)):
        raise ValueError('labeled_count must be positive and smaller than training dataset size')
    for split in ('train', 'val', 'test'):
        image_dir = dataset_root / split / 'images'
        label_dir = dataset_root / split / 'labels'
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f'Missing image/label directories for {split}')
        for img in image_dir.iterdir():
            if img.suffix.lower() in ('.jpg', '.jpeg', '.png') and not (label_dir / (img.stem + '.txt')).is_file():
                raise FileNotFoundError(f'Missing label: {label_dir / (img.stem + ".txt")}')
    # Place already approved Gemini-labeled images in teacher-only pool, not E0's labeled pool.
    approved = {}
    for path in images:
        record = approved_root / 'B' / 'train' / (path.stem + '.json')
        if record.is_file():
            doc = json.loads(record.read_text(encoding='utf-8'))
            if (doc.get('approved') is True and doc.get('source_model', '').startswith('gemini-')
                    and doc.get('annotation_source') == 'human_reviewed_vlm_candidate'
                    and doc.get('image_sha256') == sha256(path)):
                approved[path.name] = str(record.as_posix())
    rng = random.Random(seed)
    preferred = [p for p in images if p.name in approved]
    other = [p for p in images if p.name not in approved]
    rng.shuffle(other)
    # Reserve at least one spot for an already approved teacher image.
    unlabeled_count = len(images) - labeled_count
    teacher = preferred[:unlabeled_count] + other[:max(0, unlabeled_count - len(preferred))]
    teacher_names = {p.name for p in teacher}
    labeled = [p for p in images if p.name not in teacher_names]
    # If approvals exceed pool size, the remaining approved images enter the labeled pool.
    if len(labeled) != labeled_count:
        raise AssertionError('Dataset partitioning failed')
    def entry(p):
        return {'image': p.as_posix(), 'sha256': sha256(p), 'name': p.name}
    val = sorted(p for p in (dataset_root / 'val' / 'images').iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    test = sorted(p for p in (dataset_root / 'test' / 'images').iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not val or not test:
        raise ValueError('Both val and test must contain images')
    manifest = {
        'schema_version': 1, 'task': 'B', 'seed': seed,
        'experiment': 'E0 vs E1 limited-label teacher-assisted',
        'teacher_label_policy': 'E1 only; E0 never uses approved teacher pseudo-labels',
        'labeled_human_train': [entry(p) for p in labeled],
        'teacher_only_train': [{**entry(p), 'approved_record': approved.get(p.name)} for p in teacher],
        'validation': [entry(p) for p in val],
        'test': [entry(p) for p in test],
        'note': 'Synthetic test harness only. Test labels never used for training or candidate approval.'
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / 'manifest.json'
    if dest.exists():
        raise FileExistsError(f'Refusing to overwrite existing experiment manifest: {dest}')
    dest.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    missing = [p['name'] for p in manifest['teacher_only_train'] if not p['approved_record']]
    print(f'Manifest: {dest}')
    print(f'E0 labeled training examples: {len(labeled)}')
    print(f'E1 teacher-only pool: {len(teacher)} (approved: {len(teacher)-len(missing)}, pending: {len(missing)})')
    print(f'Holdouts: val={len(val)}, test={len(test)}')
    print('Teacher-only records needed:', ', '.join(missing) if missing else 'none')
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset-root', default='datasets/task_b_picking')
    p.add_argument('--approved-root', default='datasets/teacher_approved')
    p.add_argument('--out-dir', default='outputs/e0_e1_task_b')
    p.add_argument('--labeled-count', type=int, default=20)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()
    build(args.dataset_root, args.approved_root, args.out_dir, args.seed, args.labeled_count)


if __name__ == '__main__':
    main()
