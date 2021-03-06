from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
import numpy as np

import utils

def get_test_input() :
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416, 416))

    # BGR -> RGB | H W C -> C H W
    img_ = img[:, :, ::-1].transpose((2, 0, 1))
    # add batch with normalise
    img_ = img_[np.newaxis, :,:,:]/ 255.0
    # convert to float
    img_ = torch.from_numpy(img_).float()
    # convert to variable
    img_ = Variable(img_)

    return img_





def parse_cfg(file) :

    cfg_file = open(file, 'r')

    lines = cfg_file.read().split("\n")
    lines = [x for x in lines if len(x) > 0]
    lines = [x for x in lines if x[0] != '#']
    lines = [x.rstrip().lstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines :

        if line[0] == "[" :

            # block이 비어있지 않으면 추가 되어있는 것을 blocks에 넣음
            if len(block) != 0 :
                # type 들어오면 1개의 block이 완성된 것.
                # 따라서, 이전에 block에 넣어 둔것을 blocks에 추가 해주고 비워준다.
                # 새로운 type의 block을 담기위해서!
                blocks.append(block)
                block = {}

            block["type"] = line[1:-1].rstrip()

        else :
            key, val = line.split("=")
            block[key.rstrip()] = val.lstrip()

    blocks.append(block)

    return  blocks

def create_modules(blocks):
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]) :
        module = nn.Sequential()

        # convolution
        if(x['type'] == "convolutional") :
            activation = x['activation']

            try :
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters = int(x["filters"])
            paddding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])

            if paddding :
                pad = (kernel_size - 1) // 2
            else :
                pad = 0

            # convolution layer 추가
            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)

            # module name , module
            module.add_module("conv_{0}".format(index), conv)

            # batch normalization 추가
            if batch_normalize :
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            if activation == "leaky" :
                act_fn = nn.LeakyReLU(0.1, inplace=True) # inplace => input으로 들어온 값 자체를 수정.
                module.add_module("activateion_fn_{0}".format(index), act_fn)

        # upsample
        elif(x['type'] == 'upsample'):

            stride = int(x['stride'])
            upsample = nn.Upsample(scale_factor=2, mode='bilinear')
            module.add_module("upsample_{0}".format(index), upsample)

        # route
        elif(x['type'] == 'route') :
            x["layers"] = x['layers'].split(',')
            start_route = int(x['layers'][0])

            # layer에 값이 1개가 있을 수도 있음.
            try :
                end_route = int(x['layers'][1])
            except :
                end_route = 0

            if start_route > 0 :
                start_route = start_route - index
            if end_route > 0 :
                end_route = end_route -index

            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end_route < 0 :
                filters = output_filters[index + start_route] + output_filters[index + end_route]
            else :
                filters = output_filters[index + start_route]

        # shortcut == skip connection
        elif(x['type'] == 'shortcut'):
            shortcut = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), shortcut)


        # yolo
        elif(x['type'] == 'yolo') :
            mask = x['mask'].split(',')
            mask = [int(x) for x in mask]

            # mask에 해당 하는 index의 anchor만 사용
            anchors = x['anchors'].split(',')
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]

            detection = DetectionLayer(anchors)
            module.add_module("Detection_{0}".format(index), detection)


        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return  (net_info, module_list)


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()

class DetectionLayer(nn.Module) :
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors

class Darknet(nn.Module) :
    def __init__(self, cfgfile):
        super(Darknet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules(self.blocks)

    def forward(self, x, CUDA):

        # net의 정보는 불필요. 따라서 1번부터 가져옴
        modules = self.blocks[1:]

        # key = layer의 index, value = feature map
        outputs = {}

        # 0 일 때, collector가 초기화 X, 1 일 때, collector가 초기화 O, detection map concat 가능
        write = 0

        detections = None
        for i, module in enumerate(modules) :
            module_type = (module["type"])

            # convolution, upsample 이면 network에 통과
            if module_type == "convolutional" or module_type == "upsample" :
                x = self.module_list[i](x)

            # route
            elif module_type == "route" :
                layers = module["layers"]
                layers = [int(a) for a in layers]

                # 0번째 값이 양수면, 현재 index 값을 빼서 해당 layer에서 몇 번째 뒤에 있는 layer인지 찾음.
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i

                # layer 값이 1개면, output에 그냥 넣음
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]

                # layer 값이 여러 개면,
                else :

                    # 1번째 값이 양수면, 현재 index 값을 빼서 해당 layer에서 몇 번째 뒤에 있는 layer인지 찾음.
                    if (layers[1]) > 0 :
                        layers[1] = layers[1] - i

                    # route에 있는 1개의 layer를 가져와서 concat
                    feature_map1 = outputs[i + layers[0]]
                    feature_map2 = outputs[i + layers[1]]

                    x = torch.cat((feature_map1, feature_map2), 1)

            # shortcut
            elif module_type == "shortcut" :
                from_ = int(module["from"])
                x = outputs[i - 1] + outputs[i + from_]


            # yolo
            elif module_type == "yolo" :
                anchors = self.module_list[i][0].anchors

                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])

                x = x.data
                x = utils.predict_transform(x, inp_dim, anchors, num_classes, CUDA)

                # 처음으로 나오는 yolo layer에서 detection 값을 받을 때,
                if not write:
                    detections = x
                    write = 1

                # yolo layer에서 2번 째부터 detection 값을 받을 땐, 이전 값과 concat.
                else :
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

            return detections




model = Darknet("cfg/yolov3.cfg")
inp = get_test_input()

pred = model(inp, torch.cuda.is_available())
print(pred)





























