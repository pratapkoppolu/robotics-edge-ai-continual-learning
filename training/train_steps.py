
import torch


def train_step_a(model, batch, optimizer, criterion, device):
    model.train()

    images, masks = batch
    images = images.to(device)
    masks = masks.to(device)

    optimizer.zero_grad()

    predictions = model(images)
    loss = criterion(predictions, masks)

    loss.backward()
    optimizer.step()

    return {"total_loss": float(loss.item())}


def train_step_b(model, batch, optimizer, device):
    model.train()

    images, targets = batch

    images = [img.to(device) for img in images]

    targets = [
        {key: value.to(device) for key, value in target.items()}
        for target in targets
    ]

    optimizer.zero_grad()

    # Faster R-CNN returns training losses
    loss_dict = model(images, targets)

    total_loss = sum(loss_dict.values())

    total_loss.backward()
    optimizer.step()

    results = {
        key: float(value.item())
        for key, value in loss_dict.items()
    }

    results["total_loss"] = float(total_loss.item())

    return results
