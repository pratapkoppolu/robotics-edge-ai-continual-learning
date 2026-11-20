"""M6: human-gated failure evidence review, not a drift-triggered retrainer."""
import argparse
import json
from pathlib import Path


def assess(drift, outcomes, approval):
    task = drift.get('task')
    if task not in ('A', 'B') or approval.get('task') != task:
        raise ValueError('Task mismatch or missing task')
    if not approval.get('approved', False) or not approval.get('reviewer') or approval['reviewer'] == 'REVIEWER_ID':
        raise PermissionError('Human approval and genuine reviewer ID required')
    if not approval.get('root_cause_confirmed') or not approval.get('validated_new_labels'):
        raise PermissionError('Confirmed model limitation and validated new labels required')
    if not approval.get('incident_ids') or not isinstance(approval['incident_ids'], list):
        raise PermissionError('Incident IDs required for traceability')
    if not isinstance(outcomes, dict) or outcomes.get('n', 0) < 20:
        raise PermissionError('Insufficient measured task-outcome evidence (minimum 20)')
    if outcomes.get('safety_incidents', 0) > 0:
        raise PermissionError('Safety incident: escalate independently before proceeding')
    if outcomes.get('success_rate') is None or outcomes.get('minimum_success_rate') is None:
        raise PermissionError('Measured and approved success-rate threshold required')
    if not outcomes['success_rate'] < outcomes['minimum_success_rate']:
        raise PermissionError('No measured task-performance degradation')
    # Require evidence-backed degradation, not drift alone; drift is contextual evidence.
    return {'task': task, 'approved_for_training_experiment': True,
            'approved_for_deployment': False, 'drift_detected': bool(drift.get('drift_detected')),
            'reviewer': approval['reviewer'], 'incident_ids': approval['incident_ids'],
            'reason': 'Confirmed model limitation, validated labels and degraded task outcomes'}


def read_case(drift_file, outcomes_file, approval_file):
    return assess(*(json.loads(Path(p).read_text(encoding='utf-8')) for p in (drift_file, outcomes_file, approval_file)))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--drift-report', required=True); p.add_argument('--outcomes', required=True)
    p.add_argument('--approval', required=True); p.add_argument('--out', required=True)
    a=p.parse_args(); result=read_case(a.drift_report,a.outcomes,a.approval)
    out=Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
