"""Offline planner and rollback state machine. NO transport, NO robot control."""
import argparse,json
from pathlib import Path


def make_plan(candidate, fleet, canary_count=1):
    if not candidate.get('simulation_only') or candidate.get('approved_for_deployment') is not False:
        raise ValueError('Only simulation candidates can be planned in this offline tool')
    task=candidate['task']; devices=sorted([r['robot_id'] for r in fleet if r['task']==task])
    if not devices: raise ValueError(f'No robots for task {task}')
    n=min(max(1,canary_count),len(devices))
    return {'simulation_only':True,'release_authorized':False,'network_actions':0,'candidate':candidate['candidate_version'],
            'task':task,'canary':devices[:n],'remaining':devices[n:],'state':'PLANNED','rollback_to':'previous_approved_version_per_device', 'history':[]}


def advance(plan, health):
    if plan['state'] not in ('PLANNED','CANARY_RUNNING'): raise ValueError('Terminal state, cannot advance')
    if not health.get('validated') or not isinstance(health.get('latency_p95_ms'),(int,float)) or not isinstance(health.get('error_rate'),(int,float)):
        raise ValueError('Explicit measured health inputs required')
    if health['error_rate']<0 or not 0<=health['error_rate']<=1 or health['latency_p95_ms']<0: raise ValueError('Invalid health metrics')
    if health.get('safety_incidents',0)>0 or health['error_rate']>health['max_error_rate'] or health['latency_p95_ms']>health['max_latency_p95_ms']:
        plan['state']='ROLLBACK_SIMULATED';plan['history'].append({'action':'SIMULATED_ROLLBACK','reason':'failed_health_threshold'})
    else:
        plan['state']='CANARY_PASSED_SIMULATION';plan['history'].append({'action':'SIMULATED_CANARY_PASS','reason':'health_thresholds_met'})
    return plan


def main():
    a=argparse.ArgumentParser();a.add_argument('--candidate',default='outputs/step12_task_b/candidate_manifest.json');a.add_argument('--fleet',default='configs/fleet_simulated.json');a.add_argument('--out',default='outputs/step12_task_b/canary_dryrun.json');a.add_argument('--health',help='Optional JSON file containing simulated health metrics');args=a.parse_args()
    candidate=json.loads(Path(args.candidate).read_text());fleet=json.loads(Path(args.fleet).read_text());plan=make_plan(candidate,fleet)
    if args.health:plan=advance(plan,json.loads(Path(args.health).read_text()))
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(plan,indent=2));print(json.dumps(plan,indent=2))
if __name__=='__main__':main()
