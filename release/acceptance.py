"""A6: Gate candidate on old/new holdouts; simulation can NEVER be promoted."""
import argparse
import json
from pathlib import Path
import yaml


def assess(comparison, cfg):
    before, after = comparison['before'], comparison['after']
    old_before = float(before['old']['val_map50'])
    old_after = float(after['old']['val_map50'])
    new_before = float(before['new']['val_map50'])
    new_after = float(after['new']['val_map50'])
    for value in (old_before, old_after, new_before, new_after):
        if not (0.0 <= value <= 1.0):
            raise ValueError('mAP@0.50 must be within [0,1]')
    old_delta, new_delta = old_after-old_before, new_after-new_before
    gates = {
        'old_domain_retention': old_delta >= -float(cfg['max_old_map50_drop']),
        'new_domain_gain': new_delta >= float(cfg['min_new_map50_gain']),
        'old_absolute_floor': old_after >= float(cfg['min_old_map50']),
        'new_absolute_floor': new_after >= float(cfg['min_new_map50']),
    }
    technical_pass = all(gates.values())
    simulation_only = comparison.get('simulation_only') is True
    return {
        'task': cfg['task'], 'metric': 'mAP@0.50',
        'before': {'old': old_before, 'new': new_before},
        'after': {'old': old_after, 'new': new_after},
        'deltas': {'old': old_delta, 'new': new_delta},
        'gates': gates, 'technical_pass': technical_pass,
        'simulation_only': simulation_only,
        'production_release_eligible': False,
        'release_blockers': (['simulation-only candidate', 'no Jetson evidence', 'no operational safety review', 'no signed approval']
            if simulation_only else ['Jetson evidence, operational safety review and release approval not evaluated']),
        'note': 'Technical gates evaluate synthetic held-out object detection mAP only; not robot picking success.',
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/acceptance_b.yaml')
    args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    comparison = json.loads(Path(cfg['comparison_path']).read_text(encoding='utf-8'))
    report = assess(comparison, cfg)
    out = Path(cfg['output_dir']); out.mkdir(parents=True, exist_ok=True)
    (out/'acceptance.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
