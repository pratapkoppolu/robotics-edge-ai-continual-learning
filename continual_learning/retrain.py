"""Human-approved continual finetuning POC; NEVER promotes model to fleet."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset, ConcatDataset

from datasets.brushing_dataset import BrushingDataset
from datasets.picking_dataset import PickingDataset
from models.student_a import StudentA
from models.student_b import StudentB
from training.losses import BrushingLoss
from training.evaluate import evaluate_a, evaluate_b
from continual_learning.failure_analysis import read_case
from continual_learning.progressive_unfreezing import trainable_parameters


def sha256(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def select_indices(ds, names):
    lookup={p.name:i for i,p in enumerate(ds.images)}
    missing=set(names)-set(lookup)
    if missing: raise ValueError(f'Images missing from dataset: {sorted(missing)[:5]}')
    return Subset(ds, [lookup[n] for n in names])


def load_lists(config, new_ds, old_ds):
    """Both lists contain file basenames. New and replay examples cannot overlap."""
    new=json.loads(Path(config['validated_new_manifest']).read_text(encoding='utf-8'))
    rep=json.loads(Path(config['replay_manifest']).read_text(encoding='utf-8'))
    new_names=new['images']
    if not new.get('labels_validated', False):
        raise PermissionError('validated_new_manifest must confirm labels_validated: true')
    replay_records=rep['records']
    replay_names=[r['image_id'] for r in replay_records]
    if not new_names or not replay_names or set(new_names)&set(replay_names):
        raise ValueError('Require disjoint, nonempty validated new and replay images')
    if len(new_names)!=len(set(new_names)) or len(replay_names)!=len(set(replay_names)):
        raise ValueError('Duplicate images in manifests')
    # Verify recorded content hashes before admitting historical replay.
    for rec in replay_records:
        img=Path(config['historical_dataset_root'])/'train'/'images'/rec['image_id']
        if rec.get('task')!=config['task']:
            raise ValueError('Replay task mismatch')
        if sha256(img)!=rec['image_sha256']:
            raise ValueError(f'Replay image hash changed: {img}')
    return select_indices(new_ds,new_names),select_indices(old_ds,replay_names)


def collate_b(batch):return tuple(zip(*batch))


def run(path):
    cfg=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not cfg.get('enable_training',False):
        raise PermissionError('Training disabled. Explicitly set enable_training: true after validating evidence.')
    case=read_case(cfg['drift_report'],cfg['outcomes_report'],cfg['human_approval'])
    task=cfg['task']; assert task in ('A','B')
    if task!=case['task']:raise ValueError('Task mismatch')
    out=Path(cfg['output_dir'])
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Output not empty: {out}')
    if out.resolve() in [Path('outputs/task_a').resolve(),Path('outputs/task_b').resolve()]:
        raise ValueError('Cannot overwrite original baseline')
    seed=int(cfg.get('seed',42)); random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    device=torch.device('cuda' if cfg.get('device','auto')=='auto' and torch.cuda.is_available() else 'cpu' if cfg.get('device','auto')=='auto' else cfg['device'])
    if task=='A':
        mk=lambda root,split:BrushingDataset(root,split,size=(int(cfg.get('image_size',224)),)*2)
        model=StudentA(pretrained=False).to(device); criterion=BrushingLoss(); collate=None
    else:
        mk=lambda root,split:PickingDataset(root,split)
        model=StudentB(num_classes=int(cfg['num_classes']),pretrained=False).to(device)
        criterion=None;collate=collate_b
    new_ds=mk(cfg['new_dataset_root'],'train'); old_ds=mk(cfg['historical_dataset_root'],'train')
    new, replay=load_lists(cfg,new_ds,old_ds)
    train_ds=ConcatDataset([new,replay])
    val_dl=DataLoader(mk(cfg['historical_dataset_root'],'val'),batch_size=int(cfg['batch_size']),collate_fn=collate)
    train_dl=DataLoader(train_ds,batch_size=int(cfg['batch_size']),shuffle=True,collate_fn=collate)
    ckpt=torch.load(cfg['initial_checkpoint'],map_location=device,weights_only=True)
    model.load_state_dict(ckpt['model_state'],strict=True)
    out.mkdir(parents=True,exist_ok=True)
    (out/'gate.json').write_text(json.dumps(case,indent=2),encoding='utf-8')
    best=-1.0
    import mlflow
    mlflow.set_tracking_uri(cfg.get('tracking_uri','sqlite:///mlflow.db'))
    mlflow.set_experiment('robotics_continual_learning')
    with mlflow.start_run(run_name=f'task_{task}_retraining'):
        mlflow.log_params({'task':task,'seed':seed,'initial_checkpoint_sha256':sha256(cfg['initial_checkpoint']),
                           'new_count':len(new),'replay_count':len(replay),'approved_for_deployment':False})
        mlflow.log_artifact(str(out/'gate.json'))
        epoch=0
        for stage in cfg['stages']:
            groups=trainable_parameters(model,task,stage['phase'],float(stage['head_lr']),float(stage['backbone_lr']))
            optimizer=torch.optim.AdamW(groups,weight_decay=float(cfg.get('weight_decay',1e-4)))
            for _ in range(int(stage['epochs'])):
                epoch+=1;model.train(); total=0.; n=0
                for images, targets in train_dl:
                    # Keep frozen backbone batch-norm running stats unchanged.
                    if stage['phase']=='head':
                        from continual_learning.progressive_unfreezing import backbone
                        backbone(model, task).eval()
                    optimizer.zero_grad()
                    if task=='A':
                        images=images.to(device); targets=targets.to(device)
                        loss=criterion(model(images),targets); size=len(images)
                    else:
                        images=[i.to(device) for i in images]
                        targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
                        loss=sum(model(images,targets).values());size=len(images)
                    loss.backward();optimizer.step(); total+=float(loss.item())*size;n+=size
                if task=='A':
                    metrics=evaluate_a(model,val_dl,criterion,device);score=metrics['val_iou']
                else:
                    metrics=evaluate_b(model,val_dl,device,num_classes=int(cfg['num_classes']));score=metrics['val_map50']
                record={'epoch':epoch,'phase':stage['phase'],'train_loss':total/n,**metrics}
                print(json.dumps(record),flush=True)
                mlflow.log_metrics({k:float(v) for k,v in record.items() if isinstance(v,(int,float))},step=epoch)
                checkpoint={'epoch':epoch,'model_state':model.state_dict(),'optimizer_state':optimizer.state_dict(),
                            'config':cfg,'metrics':record,'approved_for_deployment':False}
                torch.save(checkpoint,out/'last.pt')
                if score>best:
                    best=score;torch.save(checkpoint,out/'best.pt')
                with (out/'metrics.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record)+'\n')
        mlflow.log_artifact(str(out/'metrics.jsonl'))
        # Candidate not registered, promoted, released or deployed.
        mlflow.log_artifact(str(out/'best.pt'))
    return best


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--config',required=True)
    args=ap.parse_args();run(args.config)
