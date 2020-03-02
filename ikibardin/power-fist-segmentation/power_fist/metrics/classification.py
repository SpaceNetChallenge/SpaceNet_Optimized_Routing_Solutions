import numpy as np
import torch
import torch.nn as nn


class Accuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> float:
        assert len(logits.size()) == 2
        assert len(labels.size()) == 1
        assert logits.size(0) == labels.size(0)
        preds = torch.argmax(logits, dim=1)
        assert preds.size() == labels.size()

        return float((preds == labels).sum()) / labels.size(0)


class FScore(nn.Module):
    def __init__(self, beta: float):
        super().__init__()
        self._f_score_numpy = FScoreNumpy(beta=beta)

    def forward(self, logits, labels) -> float:
        if logits.size() != labels.size():
            raise ValueError(f'Size mismatch: {logits.size()} vs {labels.size()}')
        y_pred = torch.sigmoid(logits.data).cpu().numpy() > 0.1
        y_true = labels.data.cpu().numpy() > 0.5
        return torch.tensor([self._f_score_numpy(y_pred, y_true)], device=torch.device('cuda'))


class F2Score(FScore):
    def __init__(self):
        super().__init__(beta=2.0)


class FScoreNumpy:
    def __init__(self, beta: float):
        self._beta = beta

    def __call__(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        :param y_pred: boolean np.ndarray of shape (num_samples, num_classes)
        :param y_true: boolean np.ndarray of shape (num_samples, num_classes)
        :return:
        """
        if y_pred.shape != y_true.shape:
            raise ValueError(f'Shape mismatch: predicted shape {y_pred.shape} vs gt shape {y_true.shape}')
        if y_pred.dtype != np.bool:
            raise TypeError(f'Expected y_pred to be of dtype `np.bool`, got `{y_pred.dtype}`')
        if y_true.dtype != np.bool:
            raise TypeError(f'Expected y_pred to be of dtype `np.bool`, got `{y_true.dtype}`')

        tp = np.logical_and(y_pred, y_true).sum(axis=1)

        tn = np.logical_and(np.logical_not(y_pred), np.logical_not(y_true)).sum(axis=1)
        fp = np.logical_and(y_pred, np.logical_not(y_true)).sum(axis=1)
        fn = np.logical_and(np.logical_not(y_pred), y_true).sum(axis=1)

        num_samples, num_classes = y_true.shape
        assert (tp + tn + fp + fn == num_classes).all()
        assert len(tp) == num_samples

        p = tp / (tp + fp)
        r = tp / (tp + fn)

        scores = (1 + self._beta ** 2) * p * r / (self._beta ** 2 * p + r)
        scores[np.isnan(scores)] = 0.0

        assert len(scores) == num_samples
        #   return scores # FIXME
        return np.mean(scores)
