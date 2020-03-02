from typing import Tuple
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision

from power_fist.n11_lovasz_losses import LovaszHingeLoss_elu


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    @staticmethod
    def where(cond, x_1, x_2):
        cond = cond.float()
        return (cond * x_1) + ((1 - cond) * x_2)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.size() != labels.size():
            raise ValueError(f'Size mismatch: {logits.size()} vs {labels.size()}')

        P = torch.sigmoid(logits)
        focal_weight = (self.where(labels > 0.5, 1 - P, P) ** self.gamma).data

        loss = F.binary_cross_entropy_with_logits(logits, labels, weight=focal_weight, size_average=self.size_average)
        return loss


class MultiLabelSoftmaxWithCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MultiLabelSoftmaxWithCrossEntropyLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(size_average=False)

    def forward(self, predictions, labels):
        assert len(predictions) == len(labels)
        n_classes = predictions.shape[1]

        all_classes = np.arange(n_classes, dtype=np.int64)
        zero_label = torch.tensor([0]).to(predictions.device)

        loss = 0
        denominator = 0
        for prediction, positives in zip(predictions, labels):
            negatives = np.setdiff1d(all_classes, positives.data.cpu().numpy(), assume_unique=True)
            negatives_tensor = torch.tensor(negatives).to(predictions.device)
            positives_tensor = (torch.tensor(positives).to(predictions.device).unsqueeze(dim=1))

            for positive in positives_tensor:
                indices = torch.cat((positive.long(), negatives_tensor))
                loss = loss + self.criterion(prediction[indices].unsqueeze(dim=0), zero_label)
                denominator += 1

        loss /= denominator

        return loss


class RobustFocalLoss2d(nn.Module):
    # assume top 10% is outliers
    def __init__(self, gamma=2, size_average=True):
        super().__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, logit, target, class_weight=None, type='sigmoid'):
        target = target.view(-1, 1).long()

        if type == 'sigmoid':
            if class_weight is None:
                class_weight = [1] * 2  # [0.5, 0.5]

            prob = F.sigmoid(logit)
            prob = prob.view(-1, 1)
            prob = torch.cat((1 - prob, prob), 1)
            select = torch.FloatTensor(len(prob), 2).zero_().cuda()
            select.scatter_(1, target, 1.0)

        elif type == 'softmax':
            B, C, H, W = logit.size()
            if class_weight is None:
                class_weight = [1] * C  # [1/C]*C

            logit = logit.permute(0, 2, 3, 1).contiguous().view(-1, C)
            prob = F.softmax(logit, 1)
            select = torch.FloatTensor(len(prob), C).zero_().cuda()
            select.scatter_(1, target, 1.0)

        class_weight = torch.FloatTensor(class_weight).cuda().view(-1, 1)
        class_weight = torch.gather(class_weight, 0, target)

        prob = (prob * select).sum(1).view(-1, 1)
        prob = torch.clamp(prob, 1e-8, 1 - 1e-8)

        focus = torch.pow((1 - prob), self.gamma)
        focus = torch.clamp(focus, 0, 2)

        batch_loss = -class_weight * focus * prob.log()

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss

        return loss


class DiceLoss(nn.Module):
    def __init__(self, per_image=False):
        super().__init__()
        self._per_image = per_image

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.size() != labels.size():
            raise ValueError(f'logits/labels size mismatch: {logits.size()} vs {labels.size()}')
        if len(logits.size()) != 4:
            raise ValueError(f'Expected input to have (N, 1, H, W) dimensions, got {logits.size()}')
        assert logits.size(1) == 1, logits.size()

        logits = torch.sigmoid(logits)
        labels = (labels > 0.5).float()

        return self._get_soft_dice_loss(logits, labels)

    def _get_soft_dice_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        assert logits.size() == labels.size()

        batch_size = logits.size()[0]
        eps = 1e-5
        if not self._per_image:
            batch_size = 1
        dice_target = labels.contiguous().view(batch_size, -1).float()
        dice_output = logits.contiguous().view(batch_size, -1)
        intersection = torch.sum(dice_output * dice_target, dim=1)
        union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
        loss = (1.0 - (2 * intersection + eps).float() / union).mean()
        assert 0.0 <= loss.item() <= 1.0, loss
        return loss


class JaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._EPS = 1e-15

    def forward(self, logits, labels):
        if logits.size() != labels.size():
            raise ValueError(f'logits/labels size mismatch: {logits.size()} vs {labels.size()}')
        if len(logits.size()) != 4:
            raise ValueError(f'Expected input to have (N, 1, H, W) dimensions, got {logits.size()}')
        assert logits.size(1) == 1, logits.size()

        batch_size = logits.size(0)

        jaccard_target = (labels > 0.5).float().view(batch_size, -1)
        jaccard_output = torch.sigmoid(logits).view(batch_size, -1)

        intersection = (jaccard_output * jaccard_target).sum(dim=1)

        union = jaccard_output.sum(dim=1) + jaccard_target.sum(dim=1) - intersection

        losses = 1.0 - (intersection + self._EPS) / (union + self._EPS)
        loss = losses.mean()
        return loss


def iou_per_image(ypred, ytrue, EMPTY=1.0, transform=None):
    intersection = ((ytrue == 1) & (ypred == 1)).float().sum()
    union = ((ytrue == 1) | (ypred == 1)).float().sum()
    if not union:
        val = EMPTY
    else:
        val = intersection / union
    if transform is not None:
        if not isinstance(val, float):
            val = val.item()
        val = transform(val)
    return val


def iou(logits, labels, mean=True, EMPTY=1.0, per_image=True, transform=None):
    logits = logits > 0
    labels = labels > 0.5
    if not per_image:
        logits, labels = (logits,), (labels,)
    vals = []
    for ypred, ytrue in zip(logits, labels):
        vals.append(iou_per_image(ypred, ytrue, EMPTY, transform))
    if mean:
        return sum(vals) / len(vals)
    else:
        return vals


class IoU(nn.Module):
    def __init__(self, mean=True, EMPTY=1.0):
        super().__init__()
        self.mean = mean
        self.EMPTY = EMPTY

    def forward(self, logits, labels):
        assert logits.size() == labels.size(), '{} vs {}'.format(logits.size(), labels.size())
        return iou(logits, labels, self.mean, self.EMPTY)


class ChannelIoUBase(nn.Module):
    def __init__(self, channel_index, mean=True, EMPTY=1.0):
        super().__init__()
        self.mean = mean
        self.EMPTY = EMPTY
        self._channel_index = channel_index

    def forward(self, logits, labels):
        assert logits.size() == labels.size(), '{} vs {}'.format(logits.size(), labels.size())
        return iou(
            logits[:, self._channel_index, :, :],
            labels[:, self._channel_index, :, :],
            self.mean,
            self.EMPTY,
        )


class IoU_0(ChannelIoUBase):
    def __init__(self):
        super().__init__(channel_index=0)


class IoU_1(ChannelIoUBase):
    def __init__(self):
        super().__init__(channel_index=1)


class IoU_2(ChannelIoUBase):
    def __init__(self):
        super().__init__(channel_index=2)


class IoU_3(ChannelIoUBase):
    def __init__(self):
        super().__init__(channel_index=3)


class IoU_4(ChannelIoUBase):
    def __init__(self):
        super().__init__(channel_index=4)


class IoU_5(ChannelIoUBase):
    def __init__(self):
        super().__init__(channel_index=5)


class IoU_6(ChannelIoUBase):
    def __init__(self):
        super().__init__(channel_index=6)


class IoU_7(ChannelIoUBase):
    def __init__(self):
        super().__init__(channel_index=7)


class AverageMeter(object):
    def __init__(self, last_k=1):
        self.last_k = last_k
        self.buffer = deque()
        self.count = 0
        self.lsum = 0.0
        self.tsum = 0.0
        self.local = 0
        self.total = 0

    def reset(self):
        self.__init__(last_k=self.last_k)

    def update(self, val):
        self.count += 1
        self.buffer.append(val)
        if len(self.buffer) > self.last_k:
            self.lsum -= self.buffer.popleft()
        self.lsum += val
        self.tsum += val
        self.local = self.lsum / len(self.buffer)
        self.total = self.tsum / self.count


class LossMixture(nn.Module):
    def __init__(self, bce_weight=0.0, jaccard_weight=0.0, dice_weight=0.0, lovasz_weight=0.0, focal_weight=0.0,
                 topology_weight=0.0):
        super().__init__()
        weights = [bce_weight, jaccard_weight, dice_weight, lovasz_weight, focal_weight, topology_weight]
        if np.equal(weights, 0.0).all():
            raise ValueError('At least one non-zero weight required')
        losses = [
            nn.BCEWithLogitsLoss,
            JaccardLoss,
            DiceLoss,
            LovaszHingeLoss_elu,
            FocalLoss,
            TopologyAwareLoss,
        ]

        self._weights = []
        self._losses = []
        for weight, loss_ in zip(weights, losses):
            if weight != 0.0:
                self._weights.append(weight)
                self._losses.append(loss_())

    def forward(self, logits, labels):
        if logits.size() != labels.size():
            raise ValueError(f'logits/labels size mismatch: {logits.size()} vs {labels.size()}')
        if len(logits.size()) != 4:
            raise ValueError(f'Expected input to have (N, 1, H, W) dimensions, got {logits.size()}')
        assert logits.size(1) == 1, logits.size()

        res = 0.0
        for weight, loss in zip(self._weights, self._losses):
            if weight != 0.0:
                res += weight * loss(logits, labels)
        return res


class WeightedMultichannelMixture(nn.Module):
    def __init__(
            self,
            bce_weight=0.0,
            jaccard_weight=0.0,
            dice_weight=0.0,
            lovasz_weight=0.0,
            channel_weights=None,
    ):
        super().__init__()
        if channel_weights is None:
            channel_weights = [1.0, 1.0, 1.0]
        self._loss = LossMixture(
            bce_weight=bce_weight,
            jaccard_weight=jaccard_weight,
            dice_weight=dice_weight,
            lovasz_weight=lovasz_weight,
        )
        self._channel_weights = channel_weights

    def forward(self, logits, labels):
        if logits.size() != labels.size():
            raise ValueError(f'logits/labels size mismatch: {logits.size()} vs {labels.size()}')
        if len(logits.size()) != 4:
            raise ValueError(f'Expected input to have (N, C, H, W) dimensions, got {logits.size()}')

        num_channels = logits.size(1)
        if num_channels != len(self._channel_weights):
            raise ValueError(
                f'Expected input to have {num_channels} channels with channels weights '
                f'{self._channel_weights}, got shape {logits.size()}'
            )

        res = 0.0
        for channel_index, weight in enumerate(self._channel_weights):
            res += weight * self._loss(
                logits[:, channel_index:channel_index + 1, :, :], labels[:, channel_index:channel_index + 1, :, :]
            )
        return res


class DeepSupervisionLoss(nn.Module):
    def __init__(
            self,
            bce_weight=0.0,
            jaccard_weight=0.0,
            dice_weight=0.0,
            lovasz_weight=0.0,
            channel_weights=None,
            deep_sup_scale=0.4,
    ):
        super().__init__()
        if channel_weights is None:
            channel_weights = [1.0, 1.0, 1.0]
        self._loss = WeightedMultichannelMixture(
            bce_weight=bce_weight,
            jaccard_weight=jaccard_weight,
            dice_weight=dice_weight,
            lovasz_weight=lovasz_weight,
            channel_weights=channel_weights,
        )
        self._deep_sup_scale = deep_sup_scale

    def forward(self, logits, labels):
        logits_main, logits_deepsup = logits
        return self._loss(logits_main, labels) + self._deep_sup_scale * self._loss(logits_deepsup, labels)


class LossBinaryDice(nn.Module):
    def __init__(self, dice_weight=1):
        super(LossBinaryDice, self).__init__()
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight

    def forward(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.dice_weight:
            smooth = 1e-5
            target = (targets == 1.0).float()
            prediction = F.sigmoid(outputs)
            dice_part = 1 - 2 * (torch.sum(prediction * target) +
                                 smooth) / (torch.sum(prediction) + torch.sum(target) + smooth)

            loss += self.dice_weight * dice_part
        return loss


class BinaryDice2ch(nn.Module):
    def __init__(self):
        super(BinaryDice2ch, self).__init__()
        self._bindice = LossBinaryDice()

    def forward(self, logits, targets):
        loss1 = self._bindice(logits[:, 0, :, :], targets[:, 0, :, :])
        loss2 = self._bindice(logits[:, 1, :, :], targets[:, 1, :, :])
        return loss1 + 2.0 * loss2


class SoftmaxKLDivLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._log_softmax = nn.LogSoftmax(dim=1)
        self._kl_div = nn.KLDivLoss(reduction='sum')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.size() != labels.size():
            raise ValueError(f'logits/labels size mismatch: {logits.size()} vs {labels.size()}')
        if len(logits.size()) != 4:
            raise ValueError(f'Expected input to have (N, С, H, W) dimensions, got {logits.size()}')
        batch_size, num_channels, h, w = labels.size()
        predicted_log_probas = self._log_softmax(logits)
        predicted_log_probas = predicted_log_probas.transpose(0, 1).reshape(num_channels, -1).transpose(0, 1)
        labels = labels.transpose(0, 1).reshape(num_channels, -1).transpose(0, 1)
        return self._kl_div(predicted_log_probas, labels) / (batch_size * h * w)


class TopologyFeaturesExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.vgg19(pretrained=True)
        # self.normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self._layer1 = backbone.features[:4]
        self._layer2 = backbone.features[4:9]
        # self._layer3 = backbone.features[9:18]
        # self.layer4 = backbone.features[26:39]
        # self.layer5 = backbone.features[39:]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self._preprocess(x)
        out1 = self._layer1(x)
        out2 = self._layer2(out1)
        # out3 = self._layer3(out2)
        return out1, out2

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        x = x.repeat(1, 3, 1, 1)
        # print(x.size())
        return x


class TopologyAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._topology_extractor = self._get_topology_extractor()
        self._l2_loss = nn.MSELoss(reduction='mean')

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if logits.size() != labels.size():
            raise ValueError(f'logits/labels size mismatch: {logits.size()} vs {labels.size()}')
        if len(logits.size()) != 4:
            raise ValueError(f'Expected input to have (N, 1, H, W) dimensions, got {logits.size()}')
        assert logits.size(1) == 1, logits.size()
        probas = torch.sigmoid(logits)

        topology_predicted = self._topology_extractor(probas)
        topology_gt = self._topology_extractor(labels)

        result = 0.0
        for features_predicted, features_gt in zip(topology_predicted, topology_gt):
            result += self._l2_loss(features_predicted, features_gt).mean(dim=0)
        return result

    @staticmethod
    def _get_topology_extractor() -> nn.Module:
        extractor = TopologyFeaturesExtractor()
        extractor.eval()
        for parameter in extractor.parameters():
            parameter.requires_grad = False
        return nn.DataParallel(extractor).cuda()


if __name__ == '__main__':
    loss = TopologyAwareLoss()
    print(loss._topology_extractor)
