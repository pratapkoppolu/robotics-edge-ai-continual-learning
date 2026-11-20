"""Task-head first, last backbone stages next, full backbone last."""

def backbone(model, task):
    if task=='A': return model.backbone
    if task=='B': return model.detector.backbone.body
    raise ValueError('Unknown task')


def trainable_parameters(model, task, phase, head_lr, backbone_lr):
    if phase not in ('head','late','full'):
        raise ValueError('phase must be head, late, full')
    if not 0 < backbone_lr <= head_lr:
        raise ValueError('Require 0 < backbone_lr <= head_lr')
    bb=backbone(model,task)
    for p in model.parameters(): p.requires_grad=True
    for p in bb.parameters(): p.requires_grad=False
    if phase=='late':
        children=list(bb.children())
        if not children: raise ValueError('No backbone stages detected')
        for layer in children[-max(1,len(children)//3):]:
            for p in layer.parameters(): p.requires_grad=True
    elif phase=='full':
        for p in bb.parameters(): p.requires_grad=True
    head_ids={id(p) for p in bb.parameters()}
    heads=[p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    backs=[p for p in bb.parameters() if p.requires_grad]
    groups=[{'params':heads,'lr':head_lr}]
    if backs:groups.append({'params':backs,'lr':backbone_lr})
    return groups
