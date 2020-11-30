# -*- coding: utf-8 -*-
"""
Created on Mon Apr 1 15:43:29 2020
@author: Sajjad Ayobbi
"""
import cv2
import numpy as np
from skimage import transform as trans

# reference facial points, a list of coordinates (x,y)
# REFERENCE_FACIAL_POINTS = [
#     [30.29459953, 51.69630051],
#     [65.53179932, 51.50139999],
#     [48.02519989, 71.73660278],
#     [33.54930115, 92.3655014],
#     [62.72990036, 92.20410156]
# ]

REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 87],
    [62.72990036, 87]
]

DEFAULT_CROP_SIZE = (96, 112)


class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))


def get_reference_facial_points(output_size=(112, 112)):

    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS)
    tmp_crop_size = np.array(DEFAULT_CROP_SIZE)

    # size_diff = max(tmp_crop_size) - tmp_crop_size
    # tmp_5pts += size_diff / 2
    # tmp_crop_size += size_diff
    # return tmp_5pts

    x_scale = output_size[0]/tmp_crop_size[0]
    y_scale = output_size[1]/tmp_crop_size[1]
    tmp_5pts[:, 0] *= x_scale
    tmp_5pts[:, 1] *= y_scale

    return tmp_5pts

