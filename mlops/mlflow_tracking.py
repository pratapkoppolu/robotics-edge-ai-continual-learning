"""MLflow run management for the robotics student training scripts.

Local-first: tracking URI file:./mlruns; set MLFLOW_TRACKING_URI and optionally
MLFLOW_S3_ENDPOINT_URL / AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY for remote
MLflow + MinIO. Never store object-storage credentials in YAML.
"""
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path


def dataset_fingerprint(root):
    """Hash the bytes of each image/annotation in a stable relative-path order.

    Use for small POC datasets. Production: hash immutable D5 manifests instead.
    """
    root = Path(root)
    digest = hashlib.sha256()
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if not files:
        raise ValueError(f'Cannot fingerprint empty dataset: {root}')
    for path in files:
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode('utf-8'))
        digest.update(b'\0')
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
    return digest.hexdigest(), len(files)


@contextmanager
def tracking_run(cfg, config_path, task_name, dataset_hash, file_count):
    """Yield an optional active MLflow module; ensure successful/failed run closing."""
    options = cfg.get('mlflow', {})
    if not options.get('enabled', True):
        yield None
        return

    try:
        import mlflow
    except ImportError as exc:
        raise RuntimeError('Install MLflow: python -m pip install mlflow') from exc

    uri = os.getenv('MLFLOW_TRACKING_URI') or options.get('tracking_uri', 'sqlite:///mlflow.db')
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(options.get('experiment_name', f'robotics_{task_name}'))

    with mlflow.start_run(run_name=options.get('run_name', f'{task_name}_supervised')):
        params = {k: v for k, v in cfg.items() if k != 'mlflow' and isinstance(v, (str, int, float, bool))}
        params.update({'training_task': task_name, 'dataset_sha256': dataset_hash,
                       'dataset_file_count': file_count})
        mlflow.log_params(params)
        mlflow.set_tags({'stage': 'supervised-baseline', 'dataset_type': 'synthetic-poc',
                         'foundation': 'MobileNetV3-Large-ImageNet1K-v1',
                         'code_status': 'locally-executed-by-user'})
        mlflow.log_artifact(str(config_path), artifact_path='configuration')
        yield mlflow


def log_epoch(mlflow_module, metrics):
    if mlflow_module is None:
        return
    epoch = int(metrics['epoch'])
    mlflow_module.log_metrics({k: float(v) for k, v in metrics.items()
                               if k != 'epoch' and isinstance(v, (int, float))}, step=epoch)


def log_final_artifacts(mlflow_module, output_dir):
    if mlflow_module is None:
        return
    output_dir = Path(output_dir)
    for name in ('best.pt', 'last.pt', 'metrics.jsonl'):
        candidate = output_dir / name
        if candidate.is_file():
            mlflow_module.log_artifact(str(candidate), artifact_path='checkpoints' if name.endswith('.pt') else 'metrics')
