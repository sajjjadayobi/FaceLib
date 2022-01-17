# -*- coding: utf-8 -*-
"""
Created on Mon Apr 1 15:43:29 2020
@author: Sajjad Ayobbi
"""
import cv2
import math
import numpy as np
from PIL import Image
from skimage import transform as trans

# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [
    [30.29459953, 51.69630051],
    [65.53179932, 51.50139999],
    [48.02519989, 71.73660278],
    [33.54930115, 87],
    [62.72990036, 87]
]

class FaceWarpException(Exception):
    def __str__(self):
        return 'In File {}:{}'.format(
            __file__, super.__str__(self))


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




#################################################################################### 


def findEuclideanDistance(source_representation, test_representation):
    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def alignment(img, left_eye, right_eye, nose):
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye
    #-----------------------
    upside_down = False
    if nose[1] < left_eye[1] or nose[1] < right_eye[1]:
        upside_down = True
    #-----------------------
    #find rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1 #rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1 #rotate inverse direction of clock

    #-----------------------
    #find length of triangle edges
    a = findEuclideanDistance(np.array(left_eye), np.array(point_3rd))
    b = findEuclideanDistance(np.array(right_eye), np.array(point_3rd))
    c = findEuclideanDistance(np.array(right_eye), np.array(left_eye))
    #-----------------------

    if b != 0 and c != 0: #this multiplication causes division by zero in cos_a calculation
        cos_a = (b*b + c*c - a*a)/(2*b*c)
        cos_a = min(1.0, max(-1.0, cos_a))
        angle = np.arccos(cos_a) #angle in radian
        angle = (angle * 180) / math.pi #radian to degree
        #-----------------------
        #rotate base image
        if direction == -1:
            angle = 90 - angle
        if upside_down == True:
            angle = angle + 90
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))

    return img
