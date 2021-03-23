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

    # 0 == cnetre x 값, 1 == cnetre y 값, 4 == confidence , 2랑 3은 width, height 인듯.
    prediction[:, :, 0] = torch.sigmoid(prediction[:, :, 0])
    prediction[:, :, 1] = torch.sigmoid(prediction[:, :, 1])
    prediction[:, :, 4] = torch.sigmoid(prediction[:, :, 4])

    # add Center offsets
    grid = np.arange(grid_size)
    a,b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    if CUDA :
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)

    prediction[:, :, 2] += x_y_offset

    # bbox크기에 anchors를 적용
    anchors = torch.FloatTensor(anchors)

    if CUDA :
        anchors = anchors.cuda()

    # anchors를 (grid_size^2 , 1) 크기가 될때까지 반복해서 (grid_size^2 , 1)의 2D tensor로 만들어준다.
    anchors = anchors.repeat(grid_size * grid_size , 1).unsqueeze(0)

    # 크기 값에 log를 씌움.( log 1 = 0 , log 0 = -무한)
    prediction[:, :, 2:4]  = torch.exp(prediction[:, :, 2:4]) * anchors

    # class score에 sigmoid 적용
    prediction[:, :, 5: 5 + num_classes] = torch.sigmoid((prediction[:, :, 5:5+num_classes]))

    # bbox는 feature map size에 맞춰져 있음. 이걸 input img size에 맞게 바꿔야함. 따라서 stride 곱해줌
    prediction[:, :, :4] *= stride

    return prediction