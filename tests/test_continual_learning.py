import pytest
import torch
from continual_learning.failure_analysis import assess
from continual_learning.replay_buffer import select_replay,manifest_fingerprint
from continual_learning.progressive_unfreezing import trainable_parameters


def test_drift_alone_never_trains():
    with pytest.raises(PermissionError):
        assess({'task':'B','drift_detected':True},{'n':100,'success_rate':0.5,'minimum_success_rate':0.9},
               {'task':'B','approved':False})


def test_reviewer_and_labels_required():
    with pytest.raises(PermissionError):
        assess({'task':'A'}, {'n':100,'success_rate':0.5,'minimum_success_rate':0.9},
               {'task':'A','approved':True,'reviewer':'REVIEWER_ID','root_cause_confirmed':True,
                'validated_new_labels':True,'incident_ids':['X']})


def test_approved_case_still_not_deployment():
    result=assess({'task':'B','drift_detected':True},
                  {'n':100,'success_rate':0.6,'minimum_success_rate':0.9,'safety_incidents':0},
                  {'task':'B','approved':True,'reviewer':'qa_01','root_cause_confirmed':True,
                   'validated_new_labels':True,'incident_ids':['case_001']})
    assert result['approved_for_training_experiment']
    assert result['approved_for_deployment'] is False


def test_replay_is_deterministic_and_stratified():
    rows=[{'image_id':f'i{i}.jpg','task':'A','stratum':str(i%3),'image_sha256':str(i)} for i in range(12)]
    a=select_replay(rows,6,42);b=select_replay(rows,6,42)
    assert a==b and len({r['stratum'] for r in a})==3
    assert manifest_fingerprint(a)==manifest_fingerprint(b)


def test_progressive_unfreeze():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone=torch.nn.Sequential(torch.nn.Linear(4,4),torch.nn.Linear(4,4),torch.nn.Linear(4,4))
            self.decoder=torch.nn.Linear(4,1)
    m=M()
    trainable_parameters(m,'A','head',1e-3,1e-4)
    assert not any(p.requires_grad for p in m.backbone.parameters())
    assert all(p.requires_grad for p in m.decoder.parameters())
    trainable_parameters(m,'A','late',1e-3,1e-4)
    assert any(p.requires_grad for p in m.backbone.parameters())
    trainable_parameters(m,'A','full',1e-3,1e-4)
    assert all(p.requires_grad for p in m.parameters())
