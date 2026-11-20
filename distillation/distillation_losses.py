"""Supervised and teacher-assisted loss terms (not logits KD for a text VLM)."""
import torch.nn.functional as F


def segmentation_assisted_loss(student_logits, ground_truth, approved_teacher_mask,
                               supervised_criterion, teacher_weight=0.2):
    if not 0 <= teacher_weight <= 1:
        raise ValueError('teacher_weight must be in [0,1]')
    supervised = supervised_criterion(student_logits, ground_truth)
    teacher = F.binary_cross_entropy_with_logits(student_logits, approved_teacher_mask)
    return supervised + teacher_weight * teacher, {'supervised': supervised.detach().item(),
                                                  'teacher': teacher.detach().item()}


def detection_assisted_loss(supervised_losses, teacher_losses, teacher_weight=0.2):
    if not 0 <= teacher_weight <= 1:
        raise ValueError('teacher_weight must be in [0,1]')
    supervised = sum(supervised_losses.values())
    teacher = sum(teacher_losses.values())
    return supervised + teacher_weight * teacher, {'supervised': supervised.detach().item(),
                                                  'teacher': teacher.detach().item()}
