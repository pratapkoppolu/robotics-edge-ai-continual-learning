"""Simulation-only Student B continual learning, old/new holdout evaluation.
Never invokes operational approval and never registers or releases a model.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import ConcatDataset, DataLoader, Subset

from datasets.picking_dataset import PickingDataset
from models.student_b import StudentB
from training.evaluate import evaluate_b
from continual_learning.progressive_unfreezing import trainable_parameters, backbone


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def collate(batch):
    return tuple(zip(*batch))


def evaluate_pair(model, old_dl, new_dl, device, num_classes):
    return {
        'old': evaluate_b(model, old_dl, device, num_classes=num_classes),
        'new': evaluate_b(model, new_dl, device, num_classes=num_classes),
    }


def run(config_path):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    if cfg.get('simulation_only') is not True:
        raise PermissionError('This runner requires simulation_only: true')
    if cfg.get('task') != 'B':
        raise ValueError('Step 11 first simulation is Task B only')
    out = Path(cfg['output_dir'])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output directory not empty: {out}')
    new_root = Path(cfg['new_dataset_root'])
    marker = new_root / 'SIMULATION_ONLY.json'
    if not marker.exists() or not json.loads(marker.read_text(encoding='utf-8')).get('simulation_only'):
        raise PermissionError('Missing simulation-only data marker')
    if Path(cfg['checkpoint']).resolve() == (out / 'candidate_simulation_only.pt').resolve():
        raise ValueError('Cannot overwrite initial checkpoint')
    seed = int(cfg.get('seed', 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() and cfg.get('device', 'auto') == 'auto' else 'cpu')
    nc = int(cfg['num_classes'])
    old_root = cfg['historical_dataset_root']
    new_ds = PickingDataset(new_root, split='train')
    old_ds = PickingDataset(old_root, split='train')
    rng = random.Random(seed)
    idx = sorted(rng.sample(range(len(old_ds)), min(int(cfg['replay_count']), len(old_ds))))
    if len(idx) == 0:
        raise ValueError('Need historical replay examples')
    data = ConcatDataset([new_ds, Subset(old_ds, idx)])
    loader = DataLoader(data, batch_size=int(cfg['batch_size']), shuffle=True, collate_fn=collate)
    old_test = DataLoader(PickingDataset(old_root, split='test'), batch_size=1, collate_fn=collate)
    new_test = DataLoader(PickingDataset(new_root, split='test'), batch_size=1, collate_fn=collate)
    model = StudentB(num_classes=nc, pretrained=False).to(device)
    checkpoint = torch.load(cfg['checkpoint'], map_location=device, weights_only=True)
    model.load_state_dict(checkpoint['model_state'], strict=True)
    before = evaluate_pair(model, old_test, new_test, device, nc)
    out.mkdir(parents=True, exist_ok=True)
    metrics = []
    import mlflow
    mlflow.set_tracking_uri(cfg.get('tracking_uri', 'sqlite:///mlflow.db'))
    mlflow.set_experiment('robotics_step11_simulation')
    with mlflow.start_run(run_name='task_B_simulated_shift_replay'):
        mlflow.set_tags({'simulation_only': 'true', 'eligible_for_deployment': 'false'})
        mlflow.log_params({'task': 'B', 'seed': seed, 'new_examples': len(new_ds),
                           'replay_examples': len(idx), 'checkpoint_sha256': digest(cfg['checkpoint'])})
        mlflow.log_metrics({'before_old_map50': before['old']['val_map50'],
                            'before_new_map50': before['new']['val_map50']}, step=0)
        step = 0
        for stage in cfg['stages']:
            groups = trainable_parameters(model, 'B', stage['phase'], float(stage['head_lr']), float(stage['backbone_lr']))
            optimizer = torch.optim.AdamW(groups, weight_decay=float(cfg.get('weight_decay', 0.0001)))
            for _ in range(int(stage['epochs'])):
                step += 1; model.train(); total = 0.0; seen = 0
                for images, targets in loader:
                    if stage['phase'] == 'head':
                        backbone(model, 'B').eval()
                    images = [x.to(device) for x in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                    optimizer.zero_grad()
                    loss = sum(model(images, targets).values())
                    loss.backward(); optimizer.step()
                    total += loss.item() * len(images); seen += len(images)
                post = evaluate_pair(model, old_test, new_test, device, nc)
                rec = {'epoch': step, 'phase': stage['phase'], 'training_loss': total / seen,
                       'old_map50': post['old']['val_map50'], 'new_map50': post['new']['val_map50']}
                print(json.dumps(rec), flush=True)
                mlflow.log_metrics({k: float(v) for k, v in rec.items() if isinstance(v, (int, float))}, step=step)
                metrics.append(rec)
        summary = {'simulation_only': True, 'approved_for_deployment': False,
                   'source_checkpoint_sha256': digest(cfg['checkpoint']),
                   'before': before, 'after': post,
                   'old_map50_delta': post['old']['val_map50'] - before['old']['val_map50'],
                   'new_map50_delta': post['new']['val_map50'] - before['new']['val_map50'],
                   'epochs': metrics,
                   'limitations': 'Synthetic brightness/color shift; labels copied from test-source geometry; no robotic success evidence'}
        (out / 'comparison.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        torch.save({'model_state': model.state_dict(), 'simulation_only': True,
                    'approved_for_deployment': False, 'config': cfg}, out / 'candidate_simulation_only.pt')
        mlflow.log_artifact(str(out / 'comparison.json'))
        # Large model checkpoint intentionally retained locally; avoid duplicate upload by default.
    return summary

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    args = p.parse_args(); run(args.config)
