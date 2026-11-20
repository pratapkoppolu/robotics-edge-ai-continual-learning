from release.canary_simulator import make_plan,advance
from release.export_feasibility import evaluate
import pytest

def test_task_isolation():
    c={'simulation_only':True,'approved_for_deployment':False,'task':'B','candidate_version':'B-candidate'}
    f=[{'task':'A','robot_id':'a1'},{'task':'B','robot_id':'b1'},{'task':'B','robot_id':'b2'}]
    p=make_plan(c,f)
    assert p['canary']==['b1'] and p['remaining']==['b2'] and not p['release_authorized']

def test_health_failure_rollback():
    p=make_plan({'simulation_only':True,'approved_for_deployment':False,'task':'B','candidate_version':'x'},[{'task':'B','robot_id':'b'}]);advance(p,{'validated':True,'latency_p95_ms':150,'max_latency_p95_ms':100,'error_rate':0.0,'max_error_rate':0.05,'safety_incidents':0})
    assert p['state']=='ROLLBACK_SIMULATED'

def test_no_health_no_promotion():
    p=make_plan({'simulation_only':True,'approved_for_deployment':False,'task':'A','candidate_version':'x'},[{'task':'A','robot_id':'a'}])
    with pytest.raises(ValueError): advance(p,{})

def test_edge_fail_closed():
    r=evaluate({'task':'B','candidate_checkpoint':'not_a_real_model.pt'})
    assert not r['edge_deployment_ready'] and not r['release_authorized']
