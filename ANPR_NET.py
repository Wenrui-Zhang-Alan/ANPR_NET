import torch
import cv2
import torchvision.models as models
import numpy as np
import torch.nn as nn



class ANPR_NET(nn.Module):
    def __init__(self):
        super(ANPR_NET, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.fc1 = nn.Linear(1000,4)


    def forward(self, x):
        x = self.resnet(x)
        x = nn.functional.relu(self.fc1(x)) 
        return x