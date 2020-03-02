import cv2
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import BCEWithLogitsLoss, NLLLoss2d, CrossEntropyLoss
import torch.nn.functional as F


def dice_round(preds, trues, t=0.5):
    preds = (preds > t).float()
    return 1 - soft_dice_loss(preds, trues, reduce=False)


def soft_dice_loss(outputs, targets, per_image=False, reduce=True):
    batch_size = outputs.size()[0]
    eps = 1e-5
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union)
    if reduce:
        loss = loss.mean()

    return loss


def soft_dice_loss_junctions(outputs, targets):
    eps = 1e-5
    if torch.sum(targets) == 0:
        return 0.
    else:
        targets = torch.flatten(targets, 0)
        outputs = torch.flatten(torch.sigmoid(outputs), 0)
        dice_target = targets[targets > 0].contiguous()
        dice_output = outputs[targets > 0].contiguous()
        intersection = torch.sum(dice_output * dice_target)
        union = torch.sum(dice_output) + torch.sum(dice_target) + eps
        loss = (1 - (2 * intersection + eps) / union)
        return loss


def jaccard(outputs, targets, per_image=False, non_empty=False, min_pixels=5):
    batch_size = outputs.size()[0]
    eps = 1e-3
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    target_sum = torch.sum(dice_target, dim=1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    losses = 1 - (intersection + eps) / (torch.sum(dice_output + dice_target, dim=1) - intersection + eps)
    if non_empty:
        assert per_image == True
        non_empty_images = 0
        sum_loss = 0
        for i in range(batch_size):
            if target_sum[i] > min_pixels:
                sum_loss += losses[i]
                non_empty_images += 1
        if non_empty_images == 0:
            return 0
        else:
            return sum_loss / non_empty_images

    return losses.mean()


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image

    def forward(self, input, target):
        return soft_dice_loss(input, target, per_image=self.per_image)


class JaccardLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, per_image=False, non_empty=False, apply_sigmoid=False,
                 min_pixels=5):
        super().__init__()
        self.size_average = size_average
        self.register_buffer('weight', weight)
        self.per_image = per_image
        self.non_empty = non_empty
        self.apply_sigmoid = apply_sigmoid
        self.min_pixels = min_pixels

    def forward(self, input, target):
        if self.apply_sigmoid:
            input = torch.sigmoid(input)
        return jaccard(input, target, per_image=self.per_image, non_empty=self.non_empty, min_pixels=self.min_pixels)


class StableBCELoss(nn.Module):
    def __init__(self):
        super(StableBCELoss, self).__init__()

    def forward(self, input, target):
        input = input.float().view(-1)
        target = target.float().view(-1)
        neg_abs = - input.abs()
        # todo check correctness
        loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
        return loss.mean()


class ComboLoss(nn.Module):
    def __init__(self, weights, per_image=False, skip_empty=True, channel_weights=np.ones((20,)), channel_losses=None):
        super().__init__()
        self.weights = weights
        self.bce = StableBCELoss()
        self.dice = DiceLoss(per_image=per_image)
        self.jaccard = JaccardLoss(per_image=per_image)
        self.focal = FocalLoss2d()
        self.mapping = {'bce': self.bce,
                        'dice': self.dice,
                        'focal': self.focal,
                        'jaccard': self.jaccard}
        self.expect_sigmoid = {'dice', 'focal', 'jaccard'}
        self.per_channel = {'dice', 'jaccard'}
        self.values = {}
        self.channel_weights = channel_weights
        self.channel_losses = channel_losses
        self.skip_empty=skip_empty

    def forward(self, outputs, targets):
        loss = 0
        weights = self.weights
        sigmoid_input = torch.sigmoid(outputs)
        for k, v in weights.items():
            if not v:
                continue
            val = 0
            if k in self.per_channel:
                channels = targets.size(1)
                for c in range(channels):
                    if not self.channel_losses or k in self.channel_losses[c]:
                        if self.skip_empty and torch.sum(targets[:, c, ...]) < 100:
                            continue
                        val += self.channel_weights[c] * self.mapping[k](
                            sigmoid_input[:, c, ...] if k in self.expect_sigmoid else outputs[:, c, ...],
                            targets[:, c, ...])

            else:
                val = self.mapping[k](sigmoid_input if k in self.expect_sigmoid else outputs, targets)

            self.values[k] = val
            loss += self.weights[k] * val
        return loss


class FocalLoss2d(nn.Module):
    def __init__(self, gamma=2, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        outputs = outputs.contiguous()
        targets = targets.contiguous()
        eps = 1e-8
        non_ignored = targets.view(-1) != self.ignore_index
        targets = targets.view(-1)[non_ignored].float()
        outputs = outputs.contiguous().view(-1)[non_ignored]
        outputs = torch.clamp(outputs, eps, 1. - eps)
        targets = torch.clamp(targets, eps, 1. - eps)
        pt = (1 - targets) * (1 - outputs) + targets * outputs
        return (-(1. - pt) ** self.gamma * torch.log(pt)).mean()


class KappaWeighedBceLoss2d(nn.Module):
    def __init__(self, y_pow=1, num_ratings=8, weight=1.):
        super().__init__()
        w_mat = np.zeros([num_ratings, num_ratings], dtype=np.int)
        w_mat += np.arange(num_ratings)
        w_mat = (w_mat - w_mat.T) ** 2
        weights = torch.from_numpy(w_mat)
        self.y_pow = y_pow
        self.num_ratings = num_ratings
        self.register_buffer('weights', weights.float())
        self.bce = BCEWithLogitsLoss(reduction='none')
        self.w = weight

    def forward(self, predictions, labels):
        predictions = torch.flatten(predictions, start_dim=2)
        labels = torch.flatten(labels, start_dim=2)
        if torch.sum(labels) == 0:
            return 0
        predictions = predictions[:, :7, ...]
        label_full_mask = labels[:, 7:8, ...]
        labels8 = torch.cat([1 - label_full_mask, labels[:, :7, ...]], dim=1)
        weights = Variable(self.weights[torch.argmax(labels8, dim=1)], requires_grad=False)
        cce = self.bce(predictions, labels[:, :7, ...])
        weights = torch.transpose(weights, 1, 2)
        weights = weights[:, 1:, ...]
        kappa_c = (weights * torch.sigmoid(predictions))
        kappa_loss = (cce * kappa_c)
        kappa_loss = torch.cat([kappa_loss[:, c:c + 1, ...][label_full_mask > 0] for c in range(7)])
        return self.w * kappa_loss.mean()


class KappaLossPytorch(nn.Module):
    def __init__(self, y_pow=1, num_ratings=5, apply_softmax=True, cce_weight=0.1, kappa_weight=1):
        super().__init__()
        w_mat = np.zeros([num_ratings, num_ratings], dtype=np.int)
        w_mat += np.arange(num_ratings)
        w_mat = (w_mat - w_mat.T) ** 2
        weights = torch.from_numpy(w_mat)
        self.y_pow = y_pow
        self.num_ratings = num_ratings
        self.apply_softmax = apply_softmax
        self.cce = CrossEntropyLoss()
        self.cce_weight = cce_weight
        self.kappa_weight = kappa_weight
        self.register_buffer('weights', weights.float())

    def forward(self, *inputs):
        predictions, labels = inputs
        predictions = predictions.transpose(0, 1)  # NCHW -> CNHW
        predictions = predictions.flatten(1)  # flatten pixels for each channel
        predictions = predictions.transpose(0, 1)  # make shape [pixels, channels]

        labels = labels.flatten(0)
        batch_size = predictions.size(0)
        cce_loss = 0
        if self.cce_weight > 0:
            cce_loss = self.cce(predictions, labels)

        labels = torch.eye(self.num_ratings)[labels].cuda().float()
        if self.apply_softmax:
            predictions = F.softmax(predictions, 1)
        weights = Variable(self.weights, requires_grad=False)
        pred_norm = predictions ** self.y_pow
        hist_rater_a = pred_norm.sum(0)
        hist_rater_b = labels.sum(0)
        conf_mat = torch.matmul(pred_norm.transpose(dim0=0, dim1=1), labels)
        nom = (weights * conf_mat).sum()
        denom = (weights * torch.matmul(torch.reshape(hist_rater_a, (self.num_ratings, 1)),
                                        torch.reshape(hist_rater_b, (1, self.num_ratings))) / batch_size).sum()
        return self.cce_weight * cce_loss + self.kappa_weight * nom / (denom + 1e-15)


class FocalLossWithDice(nn.Module):
    def __init__(self, num_classes, ignore_index=255, gamma=2, ce_weight=1., d_weight=0.1, weight=None,
                 size_average=True):
        super().__init__()
        self.num_classes = num_classes
        self.d_weight = d_weight
        self.ce_w = ce_weight
        self.gamma = gamma
        self.nll_loss = NLLLoss2d(weight, size_average, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        ce_loss = self.nll_loss((1 - F.softmax(outputs)) ** self.gamma * F.log_softmax(outputs), targets)
        d_loss = soft_dice_loss_mc(outputs, targets, self.num_classes, ignore_index=self.ignore_index)
        return self.ce_w * ce_loss + self.d_weight * d_loss


def soft_dice_loss_mc(outputs, targets, num_classes, per_image=True, only_existing_classes=False, ignore_index=255,
                      minimum_class_pixels=10, reduce_batch=True):
    batch_size = outputs.size()[0]
    eps = 1e-5
    outputs = F.softmax(outputs)

    def _soft_dice_loss(outputs, targets):
        loss = 0
        non_empty_classes = 0
        for cls in range(1, num_classes):
            non_ignored = targets.view(-1) != ignore_index
            dice_target = (targets.view(-1)[non_ignored] == cls).float()
            dice_output = outputs[:, cls].contiguous().view(-1)[non_ignored]
            intersection = (dice_output * dice_target).sum()
            if dice_target.sum() > minimum_class_pixels:
                union = dice_output.sum() + dice_target.sum() + eps
                loss += (1 - (2 * intersection + eps) / union)
                non_empty_classes += 1
        if only_existing_classes:
            loss /= (non_empty_classes + eps)
        else:
            loss /= (num_classes - 1)
        return loss

    if per_image:
        if reduce_batch:
            loss = 0
            for i in range(batch_size):
                loss += _soft_dice_loss(torch.unsqueeze(outputs[i], 0), torch.unsqueeze(targets[i], 0))
            loss /= batch_size
        else:
            loss = torch.Tensor(
                [_soft_dice_loss(torch.unsqueeze(outputs[i], 0), torch.unsqueeze(targets[i], 0)) for i in
                 range(batch_size)])
    else:
        loss = _soft_dice_loss(outputs, targets)

    return loss
