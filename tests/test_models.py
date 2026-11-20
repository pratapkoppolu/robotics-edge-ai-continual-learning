
import torch

from models.student_a import StudentA
from models.student_b import StudentB


def test_student_a_forward():
    model = StudentA(pretrained=False)
    model.eval()

    images = torch.rand(1, 3, 224, 224)

    with torch.no_grad():
        predictions = model(images)

    assert predictions.shape == (1, 1, 224, 224)
    assert torch.isfinite(predictions).all()


def test_student_b_forward():
    model = StudentB(
        num_classes=4,
        pretrained=False
    )
    model.eval()

    images = [torch.rand(3, 320, 320)]

    with torch.no_grad():
        predictions = model(images)

    assert len(predictions) == 1

    prediction = predictions[0]

    assert "boxes" in prediction
    assert "labels" in prediction
    assert "scores" in prediction

    assert prediction["boxes"].shape[1] == 4
    assert torch.isfinite(prediction["boxes"]).all()
