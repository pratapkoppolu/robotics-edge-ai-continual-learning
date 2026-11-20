"""Create UNAPPROVED VLM candidates; does not change existing teacher-approved records."""
import argparse
import json
from pathlib import Path
import yaml
from distillation.teacher_adapter import sha256
from distillation.vlm_teacher_client import invoke_vlm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/teacher_vlm.yaml')
    parser.add_argument('--task', required=True, choices=['A','B'])
    parser.add_argument('--split', default='train', choices=['train','val','test'])
    parser.add_argument('--limit', type=int, default=1)
    args = parser.parse_args()
    if args.limit < 1: raise ValueError('--limit must be positive')
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    data_root = Path(cfg['datasets'][args.task]) / args.split / 'images'
    images = sorted(p for p in data_root.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if not images: raise ValueError(f'No images under {data_root}')
    out_dir = Path(cfg['candidate_root']) / args.task / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in images[:args.limit]:
        dst = out_dir / (path.stem + '.json')
        if dst.exists(): raise FileExistsError(f'Candidate exists; will not overwrite: {dst}')
        predicted = invoke_vlm(path, cfg, args.task)
        record = {'task':args.task, 'split':args.split, 'image_path':str(path),
                  'image_sha256':sha256(path), 'source_model':cfg['model_id'],
                  'approved':False, 'objects':predicted['objects']}
        dst.write_text(json.dumps(record,indent=2),encoding='utf-8')
        print(f'UNAPPROVED candidate: {dst}')

if __name__ == '__main__':main()
