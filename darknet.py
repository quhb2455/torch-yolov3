import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

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

    































