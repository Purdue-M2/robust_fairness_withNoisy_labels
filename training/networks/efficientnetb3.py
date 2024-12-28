'''
The code is for EfficientNetB3 backbone.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from efficientnet_pytorch import EfficientNet
from utils.registry import BACKBONE


@BACKBONE.register_module(module_name="efficientnetb3")
class EfficientNetB3(nn.Module):
    def __init__(self,efficientnet_config):
        super(EfficientNetB3, self).__init__()
        self.num_classes = efficientnet_config["num_classes"]
        inc = efficientnet_config["inc"]
        self.dropout = efficientnet_config["dropout"]
        self.mode = efficientnet_config["mode"]
        # Load the EfficientNet-B3 model without pre-trained weights
        # FIXME: load the pretrained weights from online


        self.efficientnet = EfficientNet.from_pretrained('efficientnet-b3')

        self.efficientnet._fc = nn.Identity()

        if self.dropout:
            # Add dropout layer if specified
            self.dropout_layer = nn.Dropout(p=self.dropout)

        # Initialize the last_layer layer
        self.last_layer = nn.Linear(1536, self.num_classes)

        if self.mode == 'adjust_channel':
            self.adjust_channel = nn.Sequential(
                nn.Conv2d(1536, 512, 1, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
            )

    def features(self, x):
        # Extract features from the EfficientNet-B3 model
        x = self.efficientnet.extract_features(x)
        if self.mode == 'adjust_channel':
            x = self.adjust_channel(x)
        return x

    def classifier(self, x):
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)

        # Apply dropout if specified
        if self.dropout:
            x = self.dropout_layer(x)

        # Apply last_layer layer
        x = self.last_layer(x)
        return x

    def forward(self, x):
        # Extract features and apply classifier layer
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
