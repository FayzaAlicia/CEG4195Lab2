from __future__ import annotations

import torch


def _binarize(pred: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    if pred.dtype.is_floating_point:
        pred = (pred >= threshold).float()
    return pred.float()


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    pred = _binarize(pred)
    target = _binarize(target)
    intersection = (pred * target).sum().item()
    total = pred.sum().item() + target.sum().item()
    return float((2.0 * intersection + eps) / (total + eps))


def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-7) -> float:
    pred = _binarize(pred)
    target = _binarize(target)
    intersection = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - intersection
    return float((intersection + eps) / (union + eps))


class DiceBCELoss(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, target)
        probs = torch.sigmoid(logits)
        smooth = 1e-7
        intersection = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (union + smooth))
        return bce_loss + dice_loss.mean()
