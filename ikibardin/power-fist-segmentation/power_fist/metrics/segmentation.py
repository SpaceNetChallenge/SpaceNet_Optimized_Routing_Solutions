import numpy as np

import torch
import torch.nn as nn

THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def IoU(mask1, mask2):
    inter = np.sum((mask1 >= 0.5) & (mask2 >= 0.5))
    union = np.sum((mask1 >= 0.5) | (mask2 >= 0.5))
    return inter / (1e-8 + union)


def fscore(tp, fn, fp, beta=2.0):
    if tp + fn + fp < 1:
        return 1.0
    num = (1 + beta**2) * tp
    return num / (num + (beta**2) * fn + fp)


def confusion_counts(predict_mask_seq, truth_mask_seq, iou_thresh=0.5):
    predict_masks = [m for m in predict_mask_seq if np.any(m >= 0.5)]
    truth_masks = [m for m in truth_mask_seq if np.any(m >= 0.5)]

    if len(truth_masks) == 0:
        tp, fn, fp = 0.0, 0.0, float(len(predict_masks))
        return tp, fn, fp

    pred_hits = np.zeros(len(predict_masks), dtype=np.bool)  # 0 miss, 1 hit
    truth_hits = np.zeros(len(truth_masks), dtype=np.bool)  # 0 miss, 1 hit

    for p, pred_mask in enumerate(predict_masks):
        for t, truth_mask in enumerate(truth_masks):
            if IoU(pred_mask, truth_mask) > iou_thresh:
                truth_hits[t] = True
                pred_hits[p] = True

    tp = np.sum(pred_hits)
    fn = len(truth_masks) - np.sum(truth_hits)
    fp = len(predict_masks) - tp

    return tp, fn, fp


def mean_fscore(predict_mask_seq, truth_mask_seq, iou_thresholds=THRESHOLDS, beta=2.0):
    """ calculates the average FScore for the predictions in an image over
    the iou_thresholds sets.
    predict_mask_seq: list of masks of the predicted objects in the image
    truth_mask_seq: list of masks of ground-truth objects in the image
    """
    return np.mean(
        [
            fscore(tp, fn, fp, beta)
            for (tp, fn, fp
                ) in [confusion_counts(predict_mask_seq, truth_mask_seq, iou_thresh) for iou_thresh in iou_thresholds]
        ]
    )


class DiceScore(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        if logits.size() != labels.size():
            raise ValueError(f'logits/labels size mismatch: {logits.size()} vs {labels.size()}')
        if len(logits.size()) != 4:
            raise ValueError(f'Expected input to have (N, 1, H, W) dimensions, got {logits.size()}')
        assert logits.size(1) == 1, logits.size()

        logits = logits > 0
        labels = labels > 0.5
        # logits = torch.zeros_like(labels)

        scores = []
        for pred, gt in zip(logits, labels):
            scores.append(self._get_dice_score(pred, gt))
        return np.mean(scores)

    @staticmethod
    def _get_dice_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
        assert len(pred.size()) == 3, pred.size()
        assert pred.size(0) == 1, pred.size()
        assert pred.size() == gt.size()
        if pred.sum() == 0 and gt.sum() == 0:
            # print(f'Score {1.:0.3f}  Inter {0}  pred_sum {0}  gt_sum {0}')
            return 1.0
        intersection = (pred & gt).sum().float()
        dice_score = 2 * intersection / (pred.sum() + gt.sum())

        dice_score = dice_score.item()
        assert 0.0 <= dice_score <= 1.0, dice_score
        # print(f'Score {dice_score:0.3f}  Inter {intersection}  pred_sum {pred.sum()}  gt_sum {gt.sum()}')

        return dice_score


def check_metrics():
    pass


if __name__ == '__main__':
    check_metrics()
