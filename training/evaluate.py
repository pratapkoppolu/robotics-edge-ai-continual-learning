"""Validation helpers for the task-specific supervised baselines."""
import torch


@torch.no_grad()
def evaluate_a(model, loader, criterion, device):
    model.eval()
    n, loss_total, intersection_total, union_total, dice_denom = 0, 0.0, 0.0, 0.0, 0.0
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        logits = model(images)
        loss = criterion(logits, masks)
        pred = torch.sigmoid(logits) >= 0.5
        target = masks >= 0.5
        intersection_total += (pred & target).sum().item()
        union_total += (pred | target).sum().item()
        dice_denom += pred.sum().item() + target.sum().item()
        n += images.size(0)
        loss_total += loss.item() * images.size(0)
    if n == 0:
        raise ValueError('Empty Task A validation loader')
    return {
        'val_loss': loss_total / n,
        'val_iou': (intersection_total + 1e-6) / (union_total + 1e-6),
        'val_dice': (2 * intersection_total + 1e-6) / (dice_denom + 1e-6),
    }


def box_iou(box, boxes):
    """IoU of one xyxy box against N xyxy boxes."""
    lt = torch.maximum(box[:2], boxes[:, :2])
    rb = torch.minimum(box[2:], boxes[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, 0] * wh[:, 1]
    area_a = (box[2:] - box[:2]).clamp(min=0).prod()
    area_b = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0).prod(dim=1)
    return inter / (area_a + area_b - inter).clamp(min=1e-8)


@torch.no_grad()
def evaluate_b(model, loader, device, num_classes, iou_threshold=0.5):
    """Calculate per-class AP@0.50 and macro mAP@0.50; not COCO mAP@[.50:.95].

    Classes without any ground-truth objects are excluded from the macro mean.
    """
    model.eval()
    ground_truth = {cls: 0 for cls in range(1, num_classes)}
    detections = {cls: [] for cls in range(1, num_classes)}
    image_id = 0
    for images, targets in loader:
        outputs = model([im.to(device) for im in images])
        for output, target in zip(outputs, targets):
            gt_boxes = target['boxes'].cpu()
            gt_labels = target['labels'].cpu()
            for cls in range(1, num_classes):
                ground_truth[cls] += int((gt_labels == cls).sum().item())
            for box, label, score in zip(output['boxes'].cpu(), output['labels'].cpu(), output['scores'].cpu()):
                cls = int(label)
                if 1 <= cls < num_classes:
                    detections[cls].append((float(score), image_id, box))
            # Keep each image's GT and matched flags to enforce one-to-one matches
            if image_id == 0:
                all_targets = {}
            all_targets[image_id] = (gt_boxes, gt_labels)
            image_id += 1
    if image_id == 0:
        raise ValueError('Empty Task B validation loader')
    aps = []
    for cls in range(1, num_classes):
        if ground_truth[cls] == 0:
            continue
        used = {}
        tps, fps = [], []
        for score, idx, box in sorted(detections[cls], key=lambda e: e[0], reverse=True):
            gt_boxes, gt_labels = all_targets[idx]
            candidates = gt_boxes[gt_labels == cls]
            if candidates.numel() == 0:
                tps.append(0.0); fps.append(1.0); continue
            if idx not in used:
                used[idx] = set()
            ious = box_iou(box, candidates)
            if used[idx]:
                ious[list(used[idx])] = -1
            best = int(ious.argmax())
            if float(ious[best]) >= iou_threshold:
                used[idx].add(best)
                tps.append(1.0); fps.append(0.0)
            else:
                tps.append(0.0); fps.append(1.0)
        if not tps:
            aps.append(0.0); continue
        tp = torch.tensor(tps).cumsum(0)
        fp = torch.tensor(fps).cumsum(0)
        recall = tp / ground_truth[cls]
        precision = tp / (tp + fp).clamp(min=1)
        # all-point interpolated average precision
        rec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        prec = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
        for j in range(prec.numel() - 2, -1, -1):
            prec[j] = torch.maximum(prec[j], prec[j + 1])
        changes = torch.where(rec[1:] != rec[:-1])[0]
        aps.append(float(((rec[changes + 1] - rec[changes]) * prec[changes + 1]).sum()))
    return {'val_map50': sum(aps) / len(aps) if aps else 0.0,
            'val_classes_evaluated': len(aps), 'val_images': image_id}
