from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import cv2


def predict_transform(prediction, inp_dim, anchors, num_classes, CUDA=True) :

    # B, C, W, H
    batch_size = prediction.size(0)
    # input dim = 1, 3, 416, 416
    stride = inp_dim // prediction.size(2)
    grid_size = inp_dim // stride

    # number of class + ( x1 + y1 + x2 + y2 ) + confidence
    bbox_attr = 5 + num_classes
    num_anchors = len(anchors)

    # tensor 모양을 변경핻해줌
    prediction = prediction.view(batch_size, bbox_attr * num_anchors, grid_size * grid_size)
    prediction = prediction.transpose(1,2).contiguous()
    prediction = prediction.view(batch_size, grid_size * grid_size * num_anchors, bbox_attr)

    # anchor의 크기는 블록의 h, w의 속성을 따름. 따라서 stride로 anchor를 나눠서 크기를 detection map에 맞춰줌
    anchors = [ (a[0] / stride, a[1]/stride) for a in anchors]

    # 0 == cnetre x 값, 1 == cnetre y 값, 4 == confidence
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])