# Step 6 — Supervised training pipelines

Copy `training/*.py` into `robotics_ai/training/` and `configs/*.yaml` into `robotics_ai/configs/`.

Dependencies: Python, PyTorch, a matching Torchvision build, Pillow, NumPy, PyYAML. Run from the `robotics_ai` project root:

```bash
python -m training.train_a --config configs/task_a.yaml
python -m training.train_b --config configs/task_b.yaml
```

Dataset directories and earlier source files must exist. Class IDs for detection must be 0-indexed in annotation TXT files; the previously supplied loader adds 1 for Torchvision's background class. Set `num_classes` to the number of object categories plus 1.

Output: `outputs/task_a/{last.pt,best.pt,metrics.jsonl}` and `outputs/task_b/{last.pt,best.pt,metrics.jsonl}`. These are baseline supervised training pipelines; they do not implement MLflow, knowledge distillation, retraining, dataset version manifests, test set reporting, or production deployment yet.

The detection evaluator reports **mAP@0.50 for classes with ground-truth examples**, not COCO mAP@0.50:0.95. For benchmark-grade evaluation, replace this with TorchMetrics/COCOeval and a fixed dataset manifest. Checkpoint loading from untrusted `.pt` files is unsafe; only load files produced by your own trusted training system.

`pretrained: true` downloads the official weights on first use; use `false` only for offline structural smoke testing, not final model benchmarking. Metrics from random initialization are not production metrics.
