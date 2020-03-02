import torch
import torch.nn.functional as F

import numpy as np


class flip:
    FLIP_NONE=0
    FLIP_LR=1
    FLIP_FULL=2


def flip_tensor_lr(batch):
    columns = batch.data.size()[-1]
    index = torch.autograd.Variable(
        torch.LongTensor(list(reversed(range(columns)))).cuda())
    return batch.index_select(3, index)


def flip_tensor_ud(batch):
    rows = batch.data.size()[-2]
    index = torch.autograd.Variable(
        torch.LongTensor(list(reversed(range(rows)))).cuda())
    return batch.index_select(2, index)


def to_numpy(batch):
    return np.moveaxis(batch.data.cpu().numpy(), 1, -1)


def predict(model, batch, flips=flip.FLIP_NONE):
    # predict with tta on gpu
    with torch.no_grad():
        pred1 = torch.sigmoid(model(batch))

    if flips > flip.FLIP_NONE:
        with torch.no_grad():
            pred2 = flip_tensor_lr(model(flip_tensor_lr(batch)))

        masks = [pred1, pred2]

        if flips > flip.FLIP_LR:
            with torch.no_grad():
                pred3 = flip_tensor_ud(model(flip_tensor_ud(batch)))
                pred4 = flip_tensor_ud(
                    flip_tensor_lr(
                        model(flip_tensor_ud(flip_tensor_lr(batch)))))
            masks.extend([pred3, pred4])

        masks = list(map(torch.sigmoid, masks))
        new_mask = torch.mean(torch.stack(masks, 0), 0)
        return to_numpy(new_mask)

    return to_numpy(pred1)
