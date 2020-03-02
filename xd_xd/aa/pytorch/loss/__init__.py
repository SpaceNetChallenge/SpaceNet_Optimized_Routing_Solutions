import torch
import torch.nn as nn

import torch.nn.functional as F


try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse


def weight_reshape(preds, trues, weight_channel=-1, min_weight_val=0.16):
    trues_vals = trues[:, 0:weight_channel, :, :]
    preds_vals = preds[:, 0:weight_channel, :, :]
    weights_channel = trues[:, weight_channel, :, :]

    for channel in range(preds_vals.shape[1]):
        x = preds_vals[:, channel, :, :]
        out = torch.mul(x, weights_channel)
        preds_vals[:, channel, :, :] = out

    return preds_vals, trues_vals


def dice_round(preds, trues, is_average=True):
    preds = torch.round(preds)
    return dice(preds, trues, is_average=is_average)


def dice(preds, trues, weight=None, is_average=True):
    eps = 1

    num = preds.size(0)
    preds = preds.view(num, -1)
    trues = trues.view(num, -1)
    if weight is not None:
        w = torch.autograd.Variable(weight).view(num, -1)
        preds = preds * w
        trues = trues * w
    intersection = (preds * trues).sum(1)
    scores = (2. * intersection + eps) / (preds.sum(1) + trues.sum(1) + eps)

    score = scores.sum()
    if is_average:
        score /= num
    return torch.clamp(score, 0., 1.)


def focal(preds, trues, alpha=1, gamma=2, reduce=True, logits=True):
    if logits:
        BCE_loss = F.binary_cross_entropy_with_logits(preds, trues)#, reduce=False)
    else:
        BCE_loss = F.binary_cross_entropy(preds, trues)#, reduce=False)
    pt = torch.exp(-BCE_loss)
    F_loss = alpha * (1-pt)**gamma * BCE_loss
    if reduce:
        return torch.mean(F_loss)
    else:
        return F_loss


def focal_cannab(outputs, targets, gamma=2,  ignore_index=255):
    '''From cannab sn4'''
    outputs = outputs.contiguous()
    targets = targets.contiguous()
    eps = 1e-8
    non_ignored = targets.view(-1) != ignore_index
    targets = targets.view(-1)[non_ignored].float()
    outputs = outputs.contiguous().view(-1)[non_ignored]
    outputs = torch.clamp(outputs, eps, 1. - eps)
    targets = torch.clamp(targets, eps, 1. - eps)
    pt = (1 - targets) * (1 - outputs) + targets * outputs
    return (-(1. - pt) ** gamma * torch.log(pt)).mean()


def soft_dice_loss(outputs, targets, per_image=False):
    '''
    From cannab sn4
    '''

    batch_size = outputs.size()[0]
    eps = 1e-5
    if not per_image:
        batch_size = 1
    dice_target = targets.contiguous().view(batch_size, -1).float()
    dice_output = outputs.contiguous().view(batch_size, -1)
    intersection = torch.sum(dice_output * dice_target, dim=1)
    union = torch.sum(dice_output, dim=1) + torch.sum(dice_target, dim=1) + eps
    loss = (1 - (2 * intersection + eps) / union).mean()
    return loss
