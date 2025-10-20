import copy
import sys
import os
# sys.path.append('/mnt/home/zxwang22/code/Event-ReID/models/tsmbaseline')
sys.path.append(os.path.dirname(__file__))
import torch
import torch.nn as nn
from torch.nn import init
from torchvision import models
from torch.autograd import Variable
from tsm_resnet import tsmresnet50,tsmresnet34, tsmresnet18
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


class visible_differential_module(nn.Module):
    def __init__(self, arch='resnet50', seq_len = 6, alpha_weight = 1.0):
        super(visible_differential_module, self).__init__()
        self.seq_len = seq_len
        if 'resnet50' in arch:
            resnet_model = tsmresnet50(num_segments=seq_len, pretrained=True)
            resnet_model1 = tsmresnet50(num_segments=seq_len, pretrained=True)
        elif 'resnet34' in arch:
            resnet_model = tsmresnet34(num_segments=seq_len, pretrained=True)
            resnet_model1 = tsmresnet34(num_segments=seq_len, pretrained=True)
        elif 'resnet18' in arch:
            resnet_model = tsmresnet18(num_segments=seq_len, pretrained=True)
            resnet_model1 = tsmresnet18(num_segments=seq_len, pretrained=True)
        else:
            raise NotImplementedError
        self.alpha_weight = alpha_weight
        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)
        
        # implement conv1_5 and inflate weight 
        self.conv1_temp = list(resnet_model1.children())[0]
        params = [x.clone() for x in self.conv1_temp.parameters()]
        kernel_size = params[0].size()
        new_kernel_size = kernel_size[:1] + (3 * (seq_len - 1),) + kernel_size[2:]
        new_kernels = params[0].data.mean(dim=1, keepdim=True).expand(new_kernel_size).contiguous()
        self.conv1_6 = nn.Sequential(nn.Conv2d(3 * (seq_len - 1),64,kernel_size=7,stride=2,padding=3,bias=False),nn.BatchNorm2d(64),nn.ReLU(inplace=True))
        self.conv1_6[0].weight.data = new_kernels

        self.maxpool_diff = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.resnext_layer1 =nn.Sequential(*list(resnet_model1.children())[4])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])
 
        self.avg_diff = nn.AvgPool2d(kernel_size=2,stride=2)
    
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))


         

    def forward(self, x):
        B,T,C,H,W = x.shape
        assert T == 6, "seq_len = {} in data doesn't equal to 6".format(T)
        x_c6 = self.conv1_6(self.avg_diff(torch.cat([x[:,1] - x[:,0], x[:,2] - x[:,1], x[:,3] - x[:,2], x[:,4] - x[:,3], x[:,5] - x[:,4]],1).view(B,15,H,W)))
        x_diff = self.maxpool_diff(1.0/1.0*x_c6)
        
        temp_out_diff1 = x_diff 
        x_diff = self.resnext_layer1(x_diff)

        x = x.reshape([B*T,C,H,W])

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #fusion layer1
        x = self.maxpool(x)
        temp_out_diff1 = F.interpolate(temp_out_diff1, x.size()[2:])
        _,C,H,W = x.shape
        x = x.reshape([B,T,C,H,W]) + self.alpha_weight * self.alpha * temp_out_diff1.reshape([B,1,C,H,W])
        x = x.reshape([B*T,C,H,W])
        #fusion layer2
        x = self.layer1_bak(x)
        x_diff = F.interpolate(x_diff, x.size()[2:])
        _,C,H,W = x.shape
        x = x.reshape([B,T,C,H,W]) + self.alpha_weight * self.beta * x_diff.reshape([B,1,C,H,W])
        x = x.reshape([B*T,C,H,W])
        
        return x

class EventIntegralModule(nn.Module):
    def __init__(self, arch='resnet50', seq_len=6, alpha_weight = 1.0):
        super(EventIntegralModule, self).__init__()
        self.seq_len = seq_len
        if 'resnet50' in arch:
            resnet_model = tsmresnet50(num_segments=seq_len, pretrained=True)
            resnet_model1 = tsmresnet50(num_segments=seq_len, pretrained=True)
        elif 'resnet34' in arch:
            resnet_model = tsmresnet34(num_segments=seq_len, pretrained=True)
            resnet_model1 = tsmresnet34(num_segments=seq_len, pretrained=True)
        elif 'resnet18' in arch:
            resnet_model = tsmresnet18(num_segments=seq_len, pretrained=True)
            resnet_model1 = tsmresnet18(num_segments=seq_len, pretrained=True)
        else:
            raise NotImplementedError

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])

        self.event_conv = nn.Conv2d(self.conv1.out_channels*seq_len, self.conv1.out_channels, kernel_size=1, stride=1, padding=0)
        self.event_bn = nn.BatchNorm2d(self.conv1.out_channels)
        self.event_relu = nn.ReLU(inplace=True)
        self.event_conv2 = nn.Sequential(*list(resnet_model1.children())[4])

        window = torch.zeros(seq_len, self.conv1.out_channels*seq_len)
        for i in range(seq_len):
            window[i, :(i+1)*self.conv1.out_channels] = 1.
        self.register_buffer('window', window)
        self.alpha_weight = alpha_weight
        self.alpha = nn.Parameter(torch.zeros(1))

        self.event_conv.apply(weights_init_kaiming)
        self.event_bn.apply(weights_init_kaiming)

    def forward(self, x):
        # x.shape = (B,T,C,H,W)
        B, T, C, H, W = x.shape
        assert T == self.seq_len, "seq_len = {} in data doesn't equal to {} in model".format(T, self.seq_len)
        x = x.reshape([B * T, C, H, W])
        # print(f'x1 shape: {x.shape}')
        # resnet conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) #(B*T, 64, H, W)
        x = self.maxpool(x)
        # print(f'x2 shape: {x.shape}')
        # integral module
        _,C,H,W = x.shape
        int_input = x.reshape(B, 6*C, H, W) #(B, T*64, H, W)
        # print(f'int input shape1: {int_input.shape}')
        int_input = torch.permute(int_input, (0, 2, 3, 1))  #(B, H, W, T*64)
        # print(f'int input shape2: {int_input.shape}')
        int_input = int_input.unsqueeze(3) # (B, H, W, 1, T*64)
        # print(f'int input shape3: {int_input.shape}')
      
        integral_feat = torch.mul(int_input, self.window) #(B, H, W, T, T*64)
        # print(f'integral  shape1: {integral_feat.shape}')
        integral_feat = torch.permute(integral_feat, (0, 3, 4, 1, 2)) #(B, T, T*64, H, W)
        # print(f'integral  shape2: {integral_feat.shape}')
        integral_feat = integral_feat.reshape(-1, integral_feat.shape[2], integral_feat.shape[3], integral_feat.shape[4]) #(B*T, T*64, H, W)
        # print(f'integral  shape3: {integral_feat.shape}')
        integral_feat = self.event_relu(self.event_bn(self.event_conv(integral_feat))) #(B*T, conv1_out_dim, H, W)
        # print(f'integral  shape4: {integral_feat.shape}')
        integral_feat = self.event_conv2(integral_feat) # (#(B*T, layer1_hid_dim, H, W))
        # print(f'integral  shape5: {integral_feat.shape}')
        integral_feat = integral_feat.reshape(B, -1, integral_feat.shape[1], integral_feat.shape[2], integral_feat.shape[3]) #(B, T, convlayer1_hid_dim, H, W)
        # print(f'integral  shape6: {integral_feat.shape}')

        # merge the two parts
        x = self.layer1_bak(x)
        # print(f'layer1 bak shape1: {x.shape}')
        x = x.reshape(B, -1, x.shape[1], x.shape[2], x.shape[3]) #(B, T, conv2_hid_dim, H, W)
        # print(f'layer1 bak shape2: {x.shape}')
        x = x + self.alpha_weight * self.alpha * integral_feat
        _,_,C,H,W = x.shape
        x = x.reshape([B*T,C,H,W])
        return x








class DI_net(nn.Module):
    def __init__(self, class_num, seq_lenth=6, arch='resnet50', int_alpha = 1.0, dif_alpha = 1.0):
        super(DI_net, self).__init__()

        self.event_integral_module = EventIntegralModule(arch=arch,seq_len=seq_lenth, alpha_weight=int_alpha)
        self.visible_differential_module = visible_differential_module(arch=arch,seq_len=seq_lenth, alpha_weight=dif_alpha)
        if 'resnet50' in arch:
            resnet_model = tsmresnet50(num_segments= seq_lenth, pretrained=True)
        elif 'resnet34' in arch:
            resnet_model = tsmresnet34(num_segments= seq_lenth, pretrained=True)
        elif 'resnet18' in arch:
            resnet_model = tsmresnet18(num_segments= seq_lenth, pretrained=True)
        else:
            raise NotImplementedError
        self.layer2_bak = nn.Sequential(*list(resnet_model.children())[5])
        self.layer3_bak = nn.Sequential(*list(resnet_model.children())[6])
        self.layer4_bak = nn.Sequential(*list(resnet_model.children())[7])
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
 

    def forward(self, x1, x2, adj, modal=0, seq_len = 8, cpa = False):
        b, c, h, w = x1.size()
    
        x1 = x1.view(b, seq_len, int(c / seq_len), h, w)
        x2 = x2.view(b, seq_len, int(c / seq_len), h, w)
        
        if modal == 0:
            x1 = self.visible_differential_module(x1) # x1.shape = [B*T,2*C,H,W] 2 means diff and inter
            x2 = self.event_integral_module(x2) # x2.shape = [B*T,2*C,H,W] 2 means diff and inter
            # import pdb; pdb.set_trace()
            x = torch.cat((x1, x2), 0)
           
        elif modal == 1:
            x = self.visible_differential_module(x1)
        elif modal == 2:
            x = self.event_integral_module(x2)

        x = self.layer2_bak(x)
        x = self.layer3_bak(x)
        x = self.layer4_bak(x)
        x_pool = self.avgpool(x).squeeze()
        x_pool = x_pool.view(x_pool.size(0)//seq_len, seq_len, -1).mean(1) 
        
        feat = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.classifier(feat)
        else:
            return self.l2norm(feat)


