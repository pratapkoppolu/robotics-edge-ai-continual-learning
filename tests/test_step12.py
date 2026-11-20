from release.acceptance import assess
from release.hardware_readiness import readiness

CFG={'task':'B','max_old_map50_drop':.05,'min_new_map50_gain':0.,'min_old_map50':.5,'min_new_map50':.5}
def example(old_before=.8,old_after=.79,new_before=.2,new_after=.7):
    return {'simulation_only':True,'before':{'old':{'val_map50':old_before},'new':{'val_map50':new_before}},'after':{'old':{'val_map50':old_after},'new':{'val_map50':new_after}}}
def test_acceptance_pass_but_not_release():
    r=assess(example(),CFG);assert r['technical_pass'] and not r['production_release_eligible']
def test_retention_failure():
    r=assess(example(old_after=.55),CFG);assert not r['technical_pass'] and not r['gates']['old_domain_retention']
def test_negative_transfer():
    r=assess(example(new_after=.1),CFG);assert not r['gates']['new_domain_gain']
def test_simulated_never_eligible_even_with_fake_hardware_evidence():
    a=assess(example(),CFG);c={'simulation_only':True};e={k:True for k in ('actual_target_jetson_test','memory_budget_pass','latency_budget_pass','runtime_parity_pass','robot_safety_signoff','authorized_release_approval')}
    r=readiness(c,a,e);assert not r['readiness_pass'] and not r['release_authorized']
