"""A8/F6: Immutable identity for experimental candidate; no registration/promotion."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import yaml


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def build_manifest(checkpoint, comparison, acceptance, version_prefix='student-b-experimental'):
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
    if not comparison.get('simulation_only') or not acceptance.get('simulation_only'):
        raise ValueError('This registration-preparation utility only accepts simulation-only records')
    digest = sha256(checkpoint)
    return {
        'schema_version': 1, 'candidate_version': f'{version_prefix}-{digest[:12]}',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'task': 'B', 'artifact': str(checkpoint), 'artifact_sha256': digest,
        'source_checkpoint_sha256': comparison.get('source_checkpoint_sha256'),
        'evaluation_metric': 'mAP@0.50', 'technical_pass': bool(acceptance['technical_pass']),
        'simulation_only': True, 'approved_for_deployment': False,
        'release_channel': 'none', 'stage': 'experimental',
        'mlflow_registered_model_version': None,
        'note': 'Candidate identity only. Not an MLflow Registry promotion or edge package.',
    }


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/acceptance_b.yaml'); args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    out=Path(cfg['output_dir']); out.mkdir(parents=True,exist_ok=True)
    comparison=json.loads(Path(cfg['comparison_path']).read_text(encoding='utf-8'))
    acceptance=json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    m=build_manifest(cfg['candidate_checkpoint'],comparison,acceptance)
    dest=out/'candidate_manifest.json'
    if dest.exists(): raise FileExistsError(f'Preserving existing immutable candidate manifest: {dest}')
    dest.write_text(json.dumps(m,indent=2),encoding='utf-8')
    print(json.dumps(m,indent=2))

if __name__=='__main__':main()
