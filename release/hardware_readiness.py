"""A12/F3 evidence checklist; refusing simulated candidates by design."""
import argparse
import json
from pathlib import Path
import yaml


def readiness(candidate, acceptance, evidence=None):
    checks = {
      'acceptance_gates_pass': bool(acceptance.get('technical_pass')),
      'non_simulated_artifact': not candidate.get('simulation_only',True),
      'target_jetson_validation': bool(evidence and evidence.get('actual_target_jetson_test')),
      'memory_budget_pass': bool(evidence and evidence.get('memory_budget_pass')),
      'latency_budget_pass': bool(evidence and evidence.get('latency_budget_pass')),
      'runtime_parity_pass': bool(evidence and evidence.get('runtime_parity_pass')),
      'robot_safety_signoff': bool(evidence and evidence.get('robot_safety_signoff')),
      'authorized_release_approval': bool(evidence and evidence.get('authorized_release_approval')),
    }
    # Intentional fail-closed: this package does not implement release approval.
    return {'checks':checks, 'readiness_pass':all(checks.values()),
            'release_authorized':False, 'deploy_action':'none',
            'note':'No edge export, device install, signing, registry promotion, canary, or rollback is executed.'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/acceptance_b.yaml');args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    out=Path(cfg['output_dir'])
    candidate=json.loads((out/'candidate_manifest.json').read_text(encoding='utf-8'))
    acceptance=json.loads((out/'acceptance.json').read_text(encoding='utf-8'))
    report=readiness(candidate,acceptance)
    (out/'edge_readiness.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
