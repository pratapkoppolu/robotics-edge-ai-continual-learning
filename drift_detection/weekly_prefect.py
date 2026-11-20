"""M3/M4 local POC: weekly Prefect runner for existing M7 -> M5 -> M6.

The .serve runner must stay running and is NOT a separately deployed Prefect Worker.
For unattended production scheduling use a Prefect work pool and worker on an always-on server.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import yaml
from prefect import flow, task


@task(name='frozen-visual-embeddings', retries=1, retry_delay_seconds=10)
def extract_features(image_dir: str, out: str):
    from drift_detection.feature_extraction import extract
    return extract(image_dir, out)


@task(name='mmd-drift-test', retries=1, retry_delay_seconds=10)
def mmd_report(ref_npz: str, cur_npz: str, name: str, out: str, log_mlflow: bool):
    from drift_detection.detect_drift import run
    return run(ref_npz, cur_npz, name, out, log_mlflow=log_mlflow)


@task(name='evidently-diagnostics', retries=1, retry_delay_seconds=10)
def evidently_report(ref_npz: str, cur_npz: str, html: str, js: str, components: int):
    from drift_detection.evidently_report import generate
    return generate(ref_npz, cur_npz, html, js, components)


@task(name='conservative-retraining-review')
def review_decision(drift_report: dict, outcomes_path: str | None, output: str):
    from drift_detection.retraining_trigger import decide
    outcomes = json.loads(Path(outcomes_path).read_text(encoding='utf-8')) if outcomes_path else None
    result = {'task': drift_report['task'], **decide(drift_report, outcomes)}
    Path(output).write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


@flow(name='robotics-weekly-drift')
def weekly_drift(config: str = 'configs/drift_weekly.yaml'):
    conf = yaml.safe_load(Path(config).read_text(encoding='utf-8'))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    root = Path(conf['output_root']) / stamp
    root.mkdir(parents=True, exist_ok=False)
    results = {}
    for name in ('A', 'B'):
        item = conf['tasks'][name]
        taskroot = root / f'task_{name}'
        taskroot.mkdir(parents=True)
        ref_npz, cur_npz = taskroot / 'reference.npz', taskroot / 'current.npz'
        # Both directories must contain fresh, distinct images.
        extract_features(item['reference_images'], str(ref_npz))
        extract_features(item['current_images'], str(cur_npz))
        drift = mmd_report(str(ref_npz), str(cur_npz), name, str(taskroot / 'drift.json'), bool(conf.get('mlflow', True)))
        diag = evidently_report(str(ref_npz), str(cur_npz), str(taskroot / 'evidently.html'), str(taskroot / 'evidently.json'), int(conf.get('components', 6)))
        decision = review_decision(drift, item.get('outcomes_json'), str(taskroot / 'decision.json'))
        results[name] = {'mmd': drift, 'evidently': diag, 'decision': decision}
    (root / 'summary.json').write_text(json.dumps({'run_utc': stamp, 'tasks': results}, indent=2), encoding='utf-8')
    return str(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/drift_weekly.yaml')
    parser.add_argument('--serve', action='store_true', help='Schedule weekly; keep process alive')
    args = parser.parse_args()
    if args.serve:
        weekly_drift.serve(name='weekly-drift-local', cron='0 9 * * MON', timezone='Asia/Kolkata', parameters={'config': args.config})
    else:
        print(weekly_drift(config=args.config))
