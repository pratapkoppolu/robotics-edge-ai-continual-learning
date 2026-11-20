"""M6: conservative triage. No retrain-on-drift automatic decision."""
import argparse
import json
from pathlib import Path


def decide(report, outcomes=None):
    if not report.get('drift_detected'):
        return {'decision': 'monitor', 'retrain': False, 'reason': 'No statistically significant shift detected'}
    if not outcomes or outcomes.get('n', 0) < 20:
        return {'decision': 'investigate', 'retrain': False, 'reason': 'Drift flagged, but task outcomes insufficient'}
    if outcomes.get('safety_incidents', 0) > 0:
        return {'decision': 'safety_escalation', 'retrain': False, 'reason': 'Safety event: independent operational escalation required'}
    threshold = outcomes.get('minimum_success_rate')
    success = outcomes.get('success_rate')
    if threshold is None or success is None:
        return {'decision': 'investigate', 'retrain': False, 'reason': 'Missing approved success-rate threshold or observed rate'}
    if success < threshold:
        return {'decision': 'human_review_for_retraining', 'retrain': False, 'reason': 'Shift and degraded task outcomes; validate root cause, labels and approval'}
    return {'decision': 'investigate', 'retrain': False, 'reason': 'Shift without measured task-performance degradation'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--drift-report', required=True)
    p.add_argument('--outcomes', help='Optional JSON file with n, success_rate, minimum_success_rate, safety_incidents')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    d = json.loads(Path(a.drift_report).read_text())
    o = json.loads(Path(a.outcomes).read_text()) if a.outcomes else None
    result = {'task': d['task'], **decide(d, o)}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
