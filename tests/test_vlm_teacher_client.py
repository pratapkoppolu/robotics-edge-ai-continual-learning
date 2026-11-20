import json
import pytest
from distillation.vlm_teacher_client import parse_teacher_json


def test_valid_candidate():
    result=parse_teacher_json(json.dumps({'objects':[{'class_id':1,'bbox_xyxy_normalized':[.1,.2,.7,.8]}]}))
    assert len(result['objects'])==1

@pytest.mark.parametrize('bad',[
    '{"objects":[{"class_id":0,"bbox_xyxy_normalized":[0.9,0,0.2,1]}]}',
    '{"objects":[{"class_id":0,"bbox_xyxy_normalized":[0,0,2,1]}]}',
    '{"objects":[{"class_id":true,"bbox_xyxy_normalized":[0,0,1,1]}]}',
    '{"commentary":"no boxes"}'
])
def test_invalid_candidate_rejected(bad):
    with pytest.raises(ValueError):parse_teacher_json(bad)
