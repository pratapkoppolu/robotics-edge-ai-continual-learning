"""Train Task B student. Run: python -m training.train_b --config configs/task_b.yaml"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.picking_dataset import PickingDataset
from models.student_b import StudentB
from training.train_steps import train_step_b
from training.evaluate import evaluate_b
from mlops.mlflow_tracking import dataset_fingerprint, tracking_run, log_epoch, log_final_artifacts


def detection_collate(batch):
    return tuple(zip(*batch))


def run(config_path):
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    seed = int(cfg.get('seed', 42))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(cfg.get('device', 'auto') if cfg.get('device', 'auto') != 'auto'
                          else ('cuda' if torch.cuda.is_available() else 'cpu'))
    data_root = Path(cfg['dataset_root'])
    train_ds = PickingDataset(data_root, 'train')
    val_ds = PickingDataset(data_root, 'val')
    train_dl = DataLoader(train_ds, batch_size=int(cfg['batch_size']), shuffle=True,
                          num_workers=int(cfg.get('num_workers', 0)), collate_fn=detection_collate)
    val_dl = DataLoader(val_ds, batch_size=int(cfg.get('batch_size')), shuffle=False,
                        num_workers=int(cfg.get('num_workers', 0)), collate_fn=detection_collate)
    num_classes = int(cfg['num_classes'])
    model = StudentB(num_classes=num_classes, pretrained=bool(cfg.get('pretrained', True))).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']),
                                   weight_decay=float(cfg.get('weight_decay', 1e-4)))
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    data_hash, file_count = dataset_fingerprint(data_root)
    with tracking_run(cfg, config_path, "task_B_picking", data_hash, file_count) as mlflow_run:
        best = -1.0
        for epoch in range(1, int(cfg['epochs']) + 1):
            loss_sum, count = 0.0, 0
            for batch in train_dl:
                result = train_step_b(model, batch, optimizer, device)
                size = len(batch[0]); loss_sum += result['total_loss'] * size; count += size
            metrics = evaluate_b(model, val_dl, device, num_classes=num_classes)
            metrics.update(epoch=epoch, train_loss=loss_sum / count)
            print(json.dumps(metrics), flush=True)
            log_epoch(mlflow_run, metrics)
            checkpoint = dict(epoch=epoch, model_state=model.state_dict(),
                              optimizer_state=optimizer.state_dict(), config=cfg,
                              best_metric=max(best, metrics['val_map50']), metrics=metrics)
            torch.save(checkpoint, out / 'last.pt')
            if metrics['val_map50'] > best:
                best = metrics['val_map50']
                torch.save(checkpoint, out / 'best.pt')
            with (out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(metrics) + '\n')
        log_final_artifacts(mlflow_run, out)
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_b.yaml')
    run(parser.parse_args().config)
