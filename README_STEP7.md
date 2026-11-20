# Step 7: MLflow experiment tracking patch

Apply to the existing `robotics_ai` project. Paths within this patch are relative to that root.

## Files
- `mlops/mlflow_tracking.py` (NEW): records parameters, SHA-256 fingerprint of the dataset directory, epoch metrics, final checkpoint artifacts.
- `training/train_a.py` and `training/train_b.py` (REPLACE existing Step 6 versions): same training behavior plus MLflow logging.
- `configs/task_a.yaml` and `configs/task_b.yaml` (REPLACE or merge `mlflow:` stanza): local experiments by default.

## Setup (PowerShell, from robotics_ai root)
```
python -m pip install mlflow
python -m mlflow --version
```

**Preserve current checkpoints before retraining**, because the original scripts overwrite outputs/task_a/{best,last}.pt and outputs/task_b/{best,last}.pt:
```
Copy-Item .\outputs\task_a .\outputs\task_a_step6_backup -Recurse
Copy-Item .\outputs\task_b .\outputs\task_b_step6_backup -Recurse
```

For a faster tracking smoke test, temporarily set `epochs: 1` in the two YAML configs, then run:
```
python -m training.train_a --config configs/task_a.yaml
python -m training.train_b --config configs/task_b.yaml
```
The pretrained model is initialized anew (not resumed). For meaningful comparisons or continuation, implement checkpoint resume and proper artifact/version management before production use.

Inspect local tracking data via:
```
python -m mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```
Open http://127.0.0.1:5000 in your browser. Separate experiments should appear for Task A and Task B. CLI syntax may vary by installed MLflow version; use `mlflow ui --help` if needed.

## Central architecture deployment (later)
Set `$env:MLFLOW_TRACKING_URI = 'http://YOUR_MLFLOW_HOST:5000'` in the shell before training. Configure MLflow *server side* with PostgreSQL backend and an object store such as MinIO for artifacts. MinIO credentials and endpoint should be environment variables on the relevant service, not checked into configs. The local file-based setup is not the central infrastructure.

## Notes
- `dataset_sha256` fingerprints dataset file bytes and paths. It is stable while files are identical; do not interpret it as a dataset semantic version or MLflow dataset record.
- `dataset_type: synthetic-poc` applies only to this smoke-test dataset. Update when replacing with real annotated data.
- Both `.pt` files include optimizer state: they are training checkpoints, not Jetson inference artifacts.
- No Model Registry registration, stage promotion, TensorRT export, cloud deployment, drift detection, or teacher distillation is claimed in Step 7.
- Existing `metrics.jsonl` is appended to (as in Step 6), so it may include previous epochs; MLflow epoch metrics are recorded only for the current run.
- A failed training run is marked FAILED by MLflow context-manager semantics, but only final checkpoint files are uploaded upon successful loop exit.
