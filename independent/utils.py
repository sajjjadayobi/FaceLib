
import numpy as np

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 24,
    'ngpu': 4,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': False,
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

# 面部参考点的坐标
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 87],
    [62.72990036, 87]
]

# 获取面部参考点 
def get_reference_facial_points(output_size=(112, 112), crop_size=(96, 112)):

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(crop_size)

    # size_diff = max(tmp_crop_size) - tmp_crop_size
    # tmp_5pts += size_diff / 2
    # tmp_crop_size += size_diff
    # return tmp_5pts

    x_scale = output_size[0]/tmp_crop_size[0]
    y_scale = output_size[1]/tmp_crop_size[1]
    tmp_5pts[:, 0] *= x_scale
    tmp_5pts[:, 1] *= y_scale

    return tmp_5pts

import cv2

def special_draw(img, box, landmarsk, name, score=100):
    """draw a bounding box on image"""
    color = (148, 133, 0)
    tl = round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line thickness
    c1 = (int(box[0]), int(box[1]))
    c2 = (int(box[2]), int(box[3]))
    # draw bounding box
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    # draw landmark
    for land in landmarsk:
        cv2.circle(img, tuple(land.int().tolist()), 3, color, -1)
    # draw score
    score = 100-(score*100/1.4)
    score = 0 if score < 0 else score
    bar = (box[3] + 2) - (box[1] - 2)
    score_final = bar - (score*bar/100)
    #cv2.rectangle(img, (box[2] + 1, box[1] - 2 + score_final), (box[2] + (tl+5), box[3] + 2), color, -1)
    cv2.rectangle(img, (int(box[2] + 1), int(box[1] - 2 + score_final)), (int(box[2] + (tl+5)), int(box[3] + 2)), color, -1)
    # draw label
    tf = max(tl - 1, 1)  # font thickness
    t_size = cv2.getTextSize(name, 0, fontScale=tl / 3, thickness=tf)[0]
    c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
    cv2.rectangle(img, c1, c2, color, -1)  # filled

    cv2.putText(img, name, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def faces_preprocessing(faces, device):
    faces = faces.permute(0, 3, 1, 2).float()
    faces = faces.div(255).to(device)
    mu = torch.as_tensor([.5, .5, .5], dtype=faces.dtype, device=device)
    faces[:].sub_(mu[:, None, None]).div_(mu[:, None, None])
    return faces












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
