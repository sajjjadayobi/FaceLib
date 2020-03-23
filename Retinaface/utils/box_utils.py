from itertools import product as product
from math import ceil
import torch


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)

def prior_box(cfg, image_size=None):
    steps = cfg['steps']
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]
    min_sizes_ = cfg['min_sizes']
    anchors = []

    for k, f in enumerate(feature_maps):
        min_sizes = min_sizes_[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors += [cx, cy, s_kx, s_ky]

    # back to torch land
    output = torch.Tensor(anchors).view(-1, 4)
    return output


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landmark(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def nms(box, scores, thresh):
    x1 = box[:, 0]
    y1 = box[:, 1]
    x2 = box[:, 2]
    y2 = box[:, 3]
    zero = torch.tensor([0.0]).to(scores.device)

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort(descending=True)

    keep = []
    while order.shape[0] > 0:
        i = order[0]
        keep.append(i)
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.max(zero, xx2 - xx1 + 1)
        h = torch.max(zero, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = torch.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
