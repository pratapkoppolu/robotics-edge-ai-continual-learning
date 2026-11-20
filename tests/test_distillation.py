import json
import torch
import pytest
from PIL import Image
from distillation.teacher_adapter import sha256, load_record, segmentation_target, detection_target
from distillation.distillation_losses import segmentation_assisted_loss, detection_assisted_loss


def test_teacher_approval_and_hash(tmp_path):
    image = tmp_path/'test.jpg'
    Image.new('RGB',(12,12),'red').save(image)
    folder = tmp_path/'teacher'/'A'/'train'
    folder.mkdir(parents=True)
    record = {'task':'A','approved':True,'reviewer':'qa','source_model':'test',
              'image_sha256':sha256(image)}
    (folder/'test.json').write_text(json.dumps(record))
    assert load_record(tmp_path/'teacher','A','train',image)==record
    record['approved'] = False
    (folder/'test.json').write_text(json.dumps(record))
    with pytest.raises(ValueError): load_record(tmp_path/'teacher','A','train',image)


def test_detection_targets():
    record={'objects':[{'class_id':2,'bbox_xywh_normalized':[0.5,0.5,0.2,0.4]}]}
    target=detection_target(record,(100,200),4)
    assert target['labels'].tolist()==[3]
    assert target['boxes'].shape==(1,4)


def test_assisted_loss_backward():
    pred=torch.zeros(2,1,8,8,requires_grad=True)
    gt=torch.ones_like(pred); mask=torch.ones_like(pred)
    loss,_=segmentation_assisted_loss(pred,gt,mask,torch.nn.BCEWithLogitsLoss(),0.2)
    loss.backward()
    assert pred.grad is not None
    p=torch.tensor(1.,requires_grad=True)
    combined,_=detection_assisted_loss({'a':p**2},{'b':p**2},0.2)
    combined.backward()
    assert p.grad is not None
