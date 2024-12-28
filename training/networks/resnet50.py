'''
The code is for ResNet50 backbone.
'''

import os
import logging
from typing import Union
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import BACKBONE

logger = logging.getLogger(__name__)


@BACKBONE.register_module(module_name="resnet50")
class ResNet50(nn.Module):
    def __init__(self, resnet_config):
        super(ResNet50, self).__init__()


        self.num_classes = resnet_config["num_classes"]
        inc = resnet_config["inc"]
        self.dropout = resnet_config["dropout"]
        self.mode = resnet_config["mode"]

        # Define layers of the backbone
        # FIXME: download the pretrained weights from online
        resnet = torchvision.models.resnet50(pretrained=True)
        self.resnet = torch.nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if self.mode == 'adjust_channel':
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(2048, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )
            self.fc = nn.Linear(512, self.num_classes)
        else:
            self.fc = nn.Linear(2048, self.num_classes)
    def features(self, inp):
        x = self.resnet(inp)
        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        return x

    def classifier(self, features):
        x = self.avgpool(features)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, inp):
        x = self.features(inp)
        out = self.classifier(x)
        return out
