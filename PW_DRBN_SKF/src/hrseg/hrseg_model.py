import torch
import torch.nn as nn
from hrseg.hrseg_lib.models import seg_hrnet
from hrseg.hrseg_lib.config import config
from hrseg.hrseg_lib.config import update_config
import argparse
import os
from glob import glob
import numpy as np
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from hrseg.hrseg_lib.datasets.pascal_ctx import PASCALContext
from hrseg.hrseg_lib.utils.modelsummary import get_model_summary
import logging
# os.environ["CUDA_VISIBLE_DEVICES"] = '6'
# logger = logging.getLogger()
# logger.setLevel(logging.INFO)
# console = logging.StreamHandler()
# logging.getLogger('').addHandler(console)
def create_hrnet():
    args = {}
    args['cfg'] = './hrseg/hrseg_lib/pascal_ctx/seg_hrnet_w48_cls59_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml'
    args['opt'] = []
    update_config(config, args)
    if torch.__version__.startswith('1'):
        module = eval('seg_hrnet')
        module.BatchNorm2d_class = module.BatchNorm2d = torch.nn.BatchNorm2d
    model = eval(config.MODEL.NAME + '.get_seg_model')(config)
    
    pretrained_dict = torch.load('./hrseg/hrnet_w48_pascal_context_cls59_480x480.pth')
    if 'state_dict' in pretrained_dict:
        pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items()
                       if k[6:] in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('HRNet load')
      
    return model

def padtensor(input_):
    mul = 16
    h, w = input_.shape[2], input_.shape[3]
    H, W = ((h + mul) // mul) * mul, ((w + mul) // mul) * mul
    padh = H - h if h % mul != 0 else 0
    padw = W - w if w % mul != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    return input_



