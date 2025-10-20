import copy
import sys
import os
sys.path.append(os.path.dirname(__file__))
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from resnet import resnet50, resnet34, resnet18
import torch.nn.functional as F
import math

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

# #####################################################################
def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)

class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        if 'resnet50' in arch:
            model_base = resnet50(pretrained=True,
                            last_conv_stride=1, last_conv_dilation=1)
        elif 'resnet34' in arch:
            model_base = resnet34(pretrained=True,
                            last_conv_stride=1, last_conv_dilation=1)
        elif 'resnet18' in arch:
            model_base = resnet18(pretrained=True,
                            last_conv_stride=1, last_conv_dilation=1)
        else:
            raise NotImplementedError
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base
       

    def forward(self, x):
        x = self.base(x)
        return x


class embed_net(nn.Module):
    def __init__(self, class_num, arch='resnet50'):
        super(embed_net, self).__init__()

        self.thermal_module = nn.Conv2d(3,3,kernel_size=1,stride=1,padding=0)
        self.visible_module = nn.Conv2d(3,3,kernel_size=1,stride=1,padding=0)
        self.relu = nn.ReLU()
        self.syncretic_module = nn.Conv2d(3,3,kernel_size=1,stride=1,padding=0)
        self.base_resnet = base_resnet(arch=arch)
        if "resnet50" in arch:
            pool_dim = 2048
        elif "resnet34" in arch:
            pool_dim = 512
        elif "resnet18" in arch:
            pool_dim = 512
        else:
            raise NotImplementedError

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
 

    def forward(self, x1, x2, modal=0, seq_len = 8):
        b, c, h, w = x1.size()
        t = seq_len
        x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
        x2 = x2.view(int(b * seq_len), int(c / seq_len), h, w)
        
        if modal == 0:
            xv = self.visible_module(x1)
            xi = self.thermal_module(x2)    
            xa = torch.mul(xv, xi)
            xa = self.relu(xa)
            xa = self.syncretic_module(xa)
            #x = torch.cat((x1, x2), 0)
            x = torch.cat((x1, x2, xa), 0)
        elif modal == 1:
            x = x1
        elif modal == 2:
            x = x2

        x = self.base_resnet(x)
  
        x_pool = self.avgpool(x).squeeze()
        x_pool = x_pool.view(x_pool.size(0)//t, t, -1).mean(1) # (seq_len, 2 * bs, hid_dim)
        
        feat = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.classifier(feat)
        else:
            return self.l2norm(feat)


