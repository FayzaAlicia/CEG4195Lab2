import torch

from metrics import dice_score, iou_score


def test_perfect_overlap_metrics_are_one():
    pred = torch.tensor([[1, 0], [1, 1]], dtype=torch.float32)
    target = torch.tensor([[1, 0], [1, 1]], dtype=torch.float32)
    assert dice_score(pred, target) == 1.0
    assert iou_score(pred, target) == 1.0


def test_no_overlap_metrics_are_zeroish():
    pred = torch.tensor([[1, 1], [0, 0]], dtype=torch.float32)
    target = torch.tensor([[0, 0], [1, 1]], dtype=torch.float32)
    assert dice_score(pred, target) < 1e-6
    assert iou_score(pred, target) < 1e-6
