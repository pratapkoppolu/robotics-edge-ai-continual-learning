"""Train Task A student. Run: python -m training.train_a --config configs/task_a.yaml"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets.brushing_dataset import BrushingDataset
from models.student_a import StudentA
from training.losses import BrushingLoss
from training.train_steps import train_step_a
from training.evaluate import evaluate_a
from mlops.mlflow_tracking import dataset_fingerprint, tracking_run, log_epoch, log_final_artifacts


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
    img_size = int(cfg.get('image_size', 224))
    train_ds = BrushingDataset(data_root, 'train', size=(img_size, img_size))
    val_ds = BrushingDataset(data_root, 'val', size=(img_size, img_size))
    train_dl = DataLoader(train_ds, batch_size=int(cfg['batch_size']), shuffle=True,
                          num_workers=int(cfg.get('num_workers', 0)), pin_memory=device.type == 'cuda')
    val_dl = DataLoader(val_ds, batch_size=int(cfg['batch_size']), shuffle=False,
                        num_workers=int(cfg.get('num_workers', 0)))
    model = StudentA(pretrained=bool(cfg.get('pretrained', True))).to(device)
    criterion = BrushingLoss(dice_weight=float(cfg.get('dice_weight', 0.5)))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']),
                                   weight_decay=float(cfg.get('weight_decay', 1e-4)))
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    data_hash, file_count = dataset_fingerprint(data_root)
    with tracking_run(cfg, config_path, "task_A_brushing", data_hash, file_count) as mlflow_run:
        best = -1.0
        for epoch in range(1, int(cfg['epochs']) + 1):
            loss_sum, count = 0.0, 0
            for batch in train_dl:
                result = train_step_a(model, batch, optimizer, criterion, device)
                size = batch[0].size(0)
                loss_sum += result['total_loss'] * size; count += size
            metrics = evaluate_a(model, val_dl, criterion, device)
            metrics.update(epoch=epoch, train_loss=loss_sum / count)
            print(json.dumps(metrics), flush=True)
            log_epoch(mlflow_run, metrics)
            checkpoint = dict(epoch=epoch, model_state=model.state_dict(),
                              optimizer_state=optimizer.state_dict(), config=cfg,
                              best_metric=max(best, metrics['val_iou']), metrics=metrics)
            torch.save(checkpoint, out / 'last.pt')
            if metrics['val_iou'] > best:
                best = metrics['val_iou']
                torch.save(checkpoint, out / 'best.pt')
            with (out / 'metrics.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(metrics) + '\n')
        log_final_artifacts(mlflow_run, out)
    return best


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/task_a.yaml')
    run(parser.parse_args().config)
