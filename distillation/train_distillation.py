"""Teacher-assisted supervised finetuning; explicitly NOT VLM logits distillation.
Run: python -m distillation.train_distillation --config configs/distillation_a.yaml
     python -m distillation.train_distillation --config configs/distillation_b.yaml
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from datasets.brushing_dataset import BrushingDataset
from datasets.picking_dataset import PickingDataset
from models.student_a import StudentA
from models.student_b import StudentB
from training.losses import BrushingLoss
from training.evaluate import evaluate_a, evaluate_b
from mlops.mlflow_tracking import dataset_fingerprint, tracking_run, log_epoch, log_final_artifacts
from distillation.teacher_adapter import load_record, segmentation_target, detection_target
from distillation.distillation_losses import segmentation_assisted_loss, detection_assisted_loss


def load_checkpoint(model, checkpoint_path, device):
    # Checkpoints from your own trusted Step 6/7 runs only: torch.load uses pickle.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt['model_state'], strict=True)


def train(cfg, config_path):
    task = cfg['task_id'].upper()
    if task not in ('A', 'B'):
        raise ValueError('task_id must be A or B')
    random.seed(int(cfg.get('seed',42))); np.random.seed(int(cfg.get('seed',42)))
    torch.manual_seed(int(cfg.get('seed',42)))
    device_name = cfg.get('device','auto')
    device = torch.device('cuda' if device_name == 'auto' and torch.cuda.is_available()
                          else 'cpu' if device_name == 'auto' else device_name)
    root = Path(cfg['dataset_root']); output = Path(cfg['output_dir'])
    if output.resolve() in (Path('outputs/task_a').resolve(), Path('outputs/task_b').resolve()):
        raise ValueError('Use a separate distillation output_dir; do not overwrite baseline')
    output.mkdir(parents=True, exist_ok=True)
    teacher_root = Path(cfg['teacher_root']); weight = float(cfg['teacher_weight'])
    if not 0 <= weight <= 1:
        raise ValueError('teacher_weight must be in [0,1]')
    batch_size = int(cfg['batch_size']); num_classes = int(cfg.get('num_classes',4))
    if task == 'A':
        size = int(cfg.get('image_size',224))
        ds = BrushingDataset(root, 'train', size=(size,size))
        val_ds = BrushingDataset(root, 'val', size=(size,size))
        model = StudentA(pretrained=False).to(device)
        criterion = BrushingLoss(dice_weight=float(cfg.get('dice_weight', 0.5)))
        collate = None
    else:
        ds = PickingDataset(root, 'train'); val_ds = PickingDataset(root, 'val')
        model = StudentB(num_classes=num_classes, pretrained=False).to(device)
        criterion = None
        collate = lambda batch: tuple(zip(*batch))
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate)
    load_checkpoint(model, cfg['baseline_checkpoint'], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg.get('weight_decay',1e-4)))
    digest, file_count = dataset_fingerprint(root)
    best = -1.0
    # Dataset image order corresponds to sorted file list in current provided loaders.
    paths = ds.images
    # Use identity-index wrapper so shuffled batches retain correct source filenames.
    class IndexedDataset(torch.utils.data.Dataset):
        def __len__(self): return len(ds)
        def __getitem__(self, idx): return idx, ds[idx]
    def indexed_collate(batch):
        indices, samples = zip(*batch)
        if task == 'A':
            imgs, masks = zip(*samples)
            return indices, torch.stack(imgs), torch.stack(masks)
        imgs, targets = zip(*samples)
        return indices, imgs, targets
    train_loader = DataLoader(IndexedDataset(), batch_size=batch_size, shuffle=True,
                              num_workers=0, collate_fn=indexed_collate)
    with tracking_run(cfg, config_path, f'{task}_teacher_assisted', digest, file_count) as mlflow:
        if mlflow is not None:
            mlflow.set_tags({'method': 'validated-target-teacher-assistance',
                             'teacher_source': cfg.get('teacher_source', 'external-approved-vlm'),
                             'baseline_checkpoint': str(cfg['baseline_checkpoint'])})
        for epoch in range(1, int(cfg['epochs'])+1):
            model.train(); loss_total = 0.0; examples=0
            for batch in train_loader:
                indices, imgs, gt = batch
                optimizer.zero_grad()
                if task == 'A':
                    imgs, gt = imgs.to(device), gt.to(device)
                    approved = []
                    for idx in indices:
                        record = load_record(teacher_root, 'A', 'train', paths[idx])
                        approved.append(segmentation_target(record, (imgs.shape[-1],imgs.shape[-2])))
                    masks = torch.stack(approved).to(device)
                    logits = model(imgs)
                    loss, _ = segmentation_assisted_loss(logits, gt, masks, criterion, weight)
                    n = imgs.shape[0]
                else:
                    imgs = [img.to(device) for img in imgs]
                    gt = [{k:v.to(device) for k,v in target.items()} for target in gt]
                    approved = []
                    for idx in indices:
                        record = load_record(teacher_root, 'B', 'train', paths[idx])
                        with Image.open(paths[idx]) as im:
                            approved.append(detection_target(record, im.size, num_classes))
                    approved = [{k:v.to(device) for k,v in target.items()} for target in approved]
                    supervised_losses = model(imgs, gt)
                    teacher_losses = model(imgs, approved)
                    loss, _ = detection_assisted_loss(supervised_losses, teacher_losses, weight)
                    n = len(imgs)
                loss.backward(); optimizer.step()
                loss_total += float(loss.item())*n; examples += n
            if task == 'A':
                metrics = evaluate_a(model, val_loader, criterion, device)
                score = metrics['val_iou']
            else:
                metrics = evaluate_b(model, val_loader, device, num_classes=num_classes)
                score = metrics['val_map50']
            metrics.update(epoch=epoch, train_loss=loss_total/examples)
            log_epoch(mlflow, metrics)
            print(json.dumps(metrics), flush=True)
            checkpoint = {'epoch': epoch, 'model_state':model.state_dict(),
                          'optimizer_state':optimizer.state_dict(), 'config': cfg,
                          'metrics':metrics, 'best_metric':max(best,score)}
            torch.save(checkpoint, output/'last.pt')
            if score > best:
                best=score; torch.save(checkpoint, output/'best.pt')
            with (output/'metrics.jsonl').open('a', encoding='utf-8') as fh:
                fh.write(json.dumps(metrics)+'\n')
        log_final_artifacts(mlflow, output)
    return best


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    args=p.parse_args()
    with open(args.config, encoding='utf-8') as f:
        config=yaml.safe_load(f)
    train(config, args.config)
