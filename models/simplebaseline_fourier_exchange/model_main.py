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
import numpy as np


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

# Defines the new fc layer and classification layer
# |--Linear--|--bn--|--relu--|--Linear--|
class FeatureBlock(nn.Module):
    def __init__(self, input_dim, low_dim, dropout=0.5, relu=True):
        super(FeatureBlock, self).__init__()
        feat_block = []
        feat_block += [nn.Linear(input_dim, low_dim)]
        feat_block += [nn.BatchNorm1d(low_dim)]

        feat_block = nn.Sequential(*feat_block)
        feat_block.apply(weights_init_kaiming)
        self.feat_block = feat_block

    def forward(self, x):
        x = self.feat_block(x)
        return x


class ClassBlock(nn.Module):
    def __init__(self, input_dim, class_num, dropout=0.5, relu=True):
        super(ClassBlock, self).__init__()
        classifier = []
        if relu:
            classifier += [nn.LeakyReLU(0.1)]
        if dropout:
            classifier += [nn.Dropout(p=dropout)]

        classifier += [nn.Linear(input_dim, class_num)]
        classifier = nn.Sequential(*classifier)
        classifier.apply(weights_init_classifier)

        self.classifier = classifier

    def forward(self, x):
        x = self.classifier(x)
        return x


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()
        if 'resnet50' in arch:
            model_v = resnet50(pretrained=True,
                            last_conv_stride=1, last_conv_dilation=1)
        elif 'resnet34' in arch:
            model_v = resnet34(pretrained=True,
                            last_conv_stride=1, last_conv_dilation=1)
        elif 'resnet18' in arch:
            model_v = resnet18(pretrained=True,
                            last_conv_stride=1, last_conv_dilation=1)
        else:
            raise NotImplementedError
        # avg pooling to global pooling
        self.visible = model_v

    def forward(self, x):
        x = self.visible.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        if 'resnet50' in arch:
            model_t = resnet50(pretrained=True,
                            last_conv_stride=1, last_conv_dilation=1)
        elif 'resnet34' in arch:
            model_t = resnet34(pretrained=True,
                            last_conv_stride=1, last_conv_dilation=1)
        elif 'resnet18' in arch:
            model_t = resnet18(pretrained=True,
                            last_conv_stride=1, last_conv_dilation=1)
        else:
            raise NotImplementedError
        # avg pooling to global pooling
        self.thermal = model_t

    def forward(self, x):
        x = self.thermal.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class edge_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(edge_module, self).__init__()

        if 'resnet50' in arch:
            model_e = resnet50(pretrained=True,
                            last_conv_stride=1, last_conv_dilation=1)
        elif 'resnet34' in arch:
            model_e = resnet34(pretrained=True,
                            last_conv_stride=1, last_conv_dilation=1)
        elif 'resnet18' in arch:
            model_e = resnet18(pretrained=True,
                            last_conv_stride=1, last_conv_dilation=1)
        else:
            raise NotImplementedError
        # avg pooling to global pooling
        self.edge = model_e

    def forward(self, x):
        x = self.edge.conv1(x)
        x = self.edge.bn1(x)
        x = self.edge.relu(x)
        x = self.edge.maxpool(x)
        return x


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
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        # t_x = self.layer4(x)
        x = self.base.layer4(x)
        return x



class TemporalMemory(nn.Module):
    def __init__(self, feat_dim=2048, mem_size=100, margin=1, seq_len=6):
        super(TemporalMemory, self).__init__()
        self.key = nn.Parameter(torch.randn(mem_size, feat_dim))
        self.val = nn.Parameter(torch.empty(mem_size, seq_len).uniform_().cuda())
        self.lstm = nn.LSTM(feat_dim, feat_dim, 1)
        self.margin = margin
        self.S = seq_len

    def forward(self, query, val):
        query = query.reshape(query.shape[0]//self.S, self.S, -1).permute(1, 0, 2)
        h0 = torch.zeros(1, query.shape[1], query.shape[2]).cuda()
        c0 = torch.zeros(1, query.shape[1], query.shape[2]).cuda()
        if self.training: self.lstm.flatten_parameters()
        output, (hn, cn) = self.lstm(query, (h0, c0))
        query_lstm = output[-1] # [B, F]

        similarity = torch.matmul(F.normalize(query_lstm, dim=1),
                                  F.normalize(self.key.t(), dim=1))
        r_att = F.softmax(similarity, dim=1)
        read = F.softmax(torch.matmul(r_att, self.val), dim=1)

        val = val.reshape(val.shape[0]//self.S, self.S, -1)
        out = torch.bmm(read.unsqueeze(1), val).squeeze(1)
        return {'out':out, 'loss':self.loss(r_att, self.margin)}

    def loss(self, r_att, margin=1):
        topk = r_att.topk(r_att.shape[0], dim=0)[0]
        distance = topk[-1] - topk[0] + margin
        mem_trip = torch.mean(torch.max(distance, torch.zeros_like(distance)))
        return {'mem_trip':mem_trip}


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.l = nn.Conv2d(channel, channel,1)
    def forward(self, x,f):
        y = self.fc(x)
        return f* y+f

def conv1x1(conv,x):
    x = x.unsqueeze(dim=-1).unsqueeze(dim=-1)
    x = conv(x)
    x = x.squeeze()
    return x

class temporal_feat_learning(nn.Module):
    def __init__(self,  ):
        super(temporal_feat_learning, self).__init__()
        dim = 2048
        self.se_1 = SELayer(2048)
        self.se_2 = SELayer(2048)
        self.se_3 = SELayer(2048)
        self.se_4 = SELayer(2048)
        self.se_5 = SELayer(2048)
        self.se_6 = SELayer(2048)

        self.a = nn.Linear(dim, dim)
        self.b = nn.Linear(dim, dim)
        self.c = nn.Linear(dim, dim)
        self.d = nn.Linear(dim, dim)
        self.e = nn.Linear(dim, dim)
        self.f = nn.Linear(dim, dim)


    def forward(self, t_x,x,x_h):
        t1 = self.a(t_x)+x[0]
        t2 = self.b(t_x)+x[1]
        t3 = self.c(t_x)+x[2]
        t4 = self.d(t_x)+x[3]
        t5 = self.e(t_x)+x[4]
        t6 = self.f(t_x)+x[5]

        f1 = self.se_1(t1/2,x_h[0]).unsqueeze(dim=1)
        f2 = self.se_2(t2/2,x_h[1]).unsqueeze(dim=1)
        f3 = self.se_3(t3/2,x_h[2]).unsqueeze(dim=1)
        f4 = self.se_4(t4/2,x_h[3]).unsqueeze(dim=1)
        f5 = self.se_5(t5/2,x_h[4]).unsqueeze(dim=1)
        f6 = self.se_6(t6/2,x_h[5]).unsqueeze(dim=1)

        f = torch.cat((f1,f2,f3,f4,f5,f6),dim=1)
        f = f.mean(dim=1)

        return f


class modal_Classifier(nn.Module):
    def __init__(self, embed_dim, modal_class):
        super(modal_Classifier, self).__init__()
        hidden_size = 1024
        self.first_layer = nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=hidden_size, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        for layer_index in range(7):
            conv_block = nn.Sequential(
                nn.Conv1d(in_channels=hidden_size, out_channels=hidden_size // 2, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm1d(hidden_size // 2),
                nn.ReLU(inplace=True)
            )
            hidden_size = hidden_size // 2  # 512-32-8
            self.layers.append(conv_block)
        self.Liner = nn.Linear(hidden_size, modal_class)

    def forward(self, latent):
        latent = latent.unsqueeze(2)
        hidden = self.first_layer(latent)
        for i in range(7):
            hidden = self.layers[i](hidden)
        style_cls_feature = hidden.squeeze(2)
        modal_cls = self.Liner(style_cls_feature)
        if self.training:
            return modal_cls  # [batch,3]


class embed_net(nn.Module):
    def __init__(self, class_num, arch='resnet50', frame_bn=False, inference_norm=True, share_conv=False, share_bn=False,
                 share_fc=False):
        super(embed_net, self).__init__()

        self.share_conv = share_conv
        self.share_bn=share_bn
        self.share_fc = share_fc

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.thermal_module_exchange = thermal_module(arch=arch)
        self.visible_module_exchange = visible_module(arch=arch)
        # self.edge_module = edge_module(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        if "resnet50" in arch:
            pool_dim = 2048
        elif "resnet34" in arch:
            pool_dim = 512
        elif "resnet18" in arch:
            pool_dim = 512
        else:
            raise NotImplementedError

        self.frame_bn = frame_bn
        self.inference_norm = inference_norm

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.bottleneck_exchange = nn.BatchNorm1d(pool_dim)
        self.bottleneck_exchange.bias.requires_grad_(False)  # no shift
        self.bottleneck2 = nn.BatchNorm2d(pool_dim)
        self.bottleneck2.bias.requires_grad_(False)  # no shift
        self.bottleneck3 = nn.BatchNorm1d(pool_dim)
        self.bottleneck3.bias.requires_grad_(False)  # no shift
        self.bottleneck3_exchange = nn.BatchNorm1d(pool_dim)
        self.bottleneck3_exchange.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.classifier_exchange = nn.Linear(pool_dim, class_num, bias=False)
        self.local_classifier = nn.Linear(pool_dim, class_num, bias=False)
        self.local_classifier_exchange = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_exchange.apply(weights_init_kaiming)
        self.bottleneck2.apply(weights_init_kaiming)
        self.bottleneck3.apply(weights_init_kaiming)
        self.bottleneck3_exchange.apply(weights_init_kaiming)

        self.classifier.apply(weights_init_classifier)
        self.classifier_exchange.apply(weights_init_classifier)
        self.local_classifier.apply(weights_init_classifier)
        self.local_classifier_exchange.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.logit_scale_dense = nn.Parameter(torch.ones([]))
        nn.init.constant_(self.logit_scale_dense, np.log(1 / 0.07))

    @staticmethod
    def sep_bn_func(input_all, bn_raw, bn_exchange):
        N_track = input_all.shape[0]
        rgb_input, rgb_exchange_input, event_input, event_exchange_input = torch.split(input_all, N_track // 4)
        raw_input = torch.cat([rgb_input, event_input], 0)
        exchange_input = torch.cat([rgb_exchange_input, event_exchange_input], 0)
        feat_raw = bn_raw(raw_input)
        feat_exchange = bn_exchange(exchange_input)
        rgb_raw_feat, event_raw_feat = torch.split(feat_raw, N_track // 4)
        rgb_exchange_feat, event_exchange_feat = torch.split(feat_exchange, N_track // 4)
        feat = torch.cat([rgb_raw_feat, rgb_exchange_feat, event_raw_feat, event_exchange_feat], 0)
        return feat

    def forward(self, x1, x2, exchanged_rgb=None, exchanged_event=None, modal=0, seq_len=8, frame_id_loss=False,
                return_raw_frame=False, gem_pooling=False):
        b, c, h, w = x1.size()
        t = seq_len
        x1 = x1.view(int(b * seq_len), int(c / seq_len), h, w)
        x2 = x2.view(int(b * seq_len), int(c / seq_len), h, w)
        if modal == 0:
            rgb_raw_out = self.visible_module(x1)
            event_raw_out = self.thermal_module(x2)
            if (exchanged_rgb is not None) and (exchanged_event is not None):
                exchanged_rgb = exchanged_rgb.view(int(b * seq_len), int(c / seq_len), h, w)
                exchanged_event = exchanged_event.view(int(b * seq_len), int(c / seq_len), h, w)
                if self.share_conv:
                    rgb_exchange_out = self.visible_module(exchanged_rgb)
                    event_exchange_out = self.thermal_module(exchanged_event)
                else:
                    rgb_exchange_out = self.visible_module_exchange(exchanged_rgb)
                    event_exchange_out = self.thermal_module_exchange(exchanged_event)
                x = torch.cat([rgb_raw_out, rgb_exchange_out, event_raw_out, event_exchange_out], 0)
            else:
                x = torch.cat([rgb_raw_out, event_raw_out], 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)

        x = self.base_resnet(x) # (num_id*num_pos*3*seq_len*2, c, h, w)
        _, c, h, w = x.shape

        x_frame_patch = x.view(-1, t, c, h, w) #(num_id*num_pos*3*2, seq_len, c, h, w)
        x_pool = self.avgpool(x).squeeze() # (num_id*num_pos*2*seq_len*2, c)
        x_frame_pool = x_pool.view(-1, t, c) #(num_id*2*num_pos*2, seq_len, c)
        x_pool = x_pool.view(x_pool.size(0)//t, t, -1) # (num_id*num_pos*2*2, seq_len, hid_dim)

        if gem_pooling:
            p = 3.0
            x_pool = (torch.mean(x_pool ** p, dim=1) + 1e-12) ** (1 / p)
        else:
            x_pool = torch.mean(x_pool, dim=1)
        if self.training and (modal == 0) and (not self.share_bn):
            feat = embed_net.sep_bn_func(x_pool, self.bottleneck, self.bottleneck_exchange)
        else:
            feat = self.bottleneck(x_pool)
        num_track = feat.shape[0]

        # reshape for frame feature BN
        if self.frame_bn:
            x_frame_pool_feat = x_frame_pool.view(-1, c)
            if self.training and (modal == 0) and (not self.share_bn):
                x_frame_pool_feat = embed_net.sep_bn_func(x_frame_pool_feat, self.bottleneck3, self.bottleneck3_exchange)
            else:
                x_frame_pool_feat = self.bottleneck3(x_frame_pool_feat)
            x_frame_pool_feat = x_frame_pool_feat.view(-1, t, c)

        if return_raw_frame:
            x_frame_return = x_frame_pool
        else:
            x_frame_return = x_frame_pool_feat

        if self.training:

            feat_rgb, feat_rgb_exchanged, feat_event, feat_event_exchanged = feat[:num_track // 4], \
                feat[num_track // 4:num_track // 2], feat[num_track // 2:num_track * 3 // 4], feat[num_track * 3 // 4:]
            feat_original = torch.cat([feat_rgb, feat_event], dim=0)
            feat_exchanged = torch.cat([feat_rgb_exchanged, feat_event_exchanged], dim=0)

            x_frame_pool_feat_rgb, x_frame_pool_feat_rgb_exchanged, x_frame_pool_feat_event, x_frame_pool_event_exchanged = \
                x_frame_pool_feat[:num_track // 4], x_frame_pool_feat[num_track // 4:num_track // 2], \
                    x_frame_pool_feat[num_track // 2:num_track * 3 // 4], x_frame_pool_feat[num_track * 3 // 4:]
            x_frame_pool_feat_original = torch.cat([x_frame_pool_feat_rgb, x_frame_pool_feat_event], dim=0)
            x_frame_pool_feat_exchanged = torch.cat([x_frame_pool_feat_rgb_exchanged, x_frame_pool_event_exchanged],
                                                    dim=0)

            x_pool_rgb, x_pool_rgb_exchange, x_pool_event, x_pool_event_exchanged = x_pool[:num_track // 4], \
                x_pool[num_track // 4:num_track // 2], x_pool[num_track // 2:num_track * 3 // 4], x_pool[num_track * 3 // 4:]
            x_pool_original = torch.cat([x_pool_rgb, x_pool_event], dim=0)
            x_pool_exchanged = torch.cat([x_pool_rgb_exchange, x_pool_event_exchanged], dim=0)

            x_frame_rgb, x_frame_rgb_exchange, x_frame_event, x_frame_event_exchanged = x_frame_return[:num_track // 4], \
                x_frame_return[num_track // 4:num_track // 2], x_frame_return[num_track // 2:num_track * 3 // 4], \
                x_frame_return[num_track * 3 // 4:]
            x_frame_original = torch.cat([x_frame_rgb, x_frame_event], dim=0)
            x_frame_exchanged = torch.cat([x_frame_rgb_exchange, x_frame_event_exchanged], dim=0)

            x_track_logits = self.classifier(feat_original)
            x_frame_logits = self.local_classifier(x_frame_pool_feat_original.reshape(-1, c))
            if self.share_fc:
                x_exchange_track_logits = self.classifier(feat_original)
                x_frame_exchange_logits = self.local_classifier(x_frame_pool_feat_original.reshape(-1, c))
            else:
                x_exchange_track_logits = self.classifier_exchange(feat_exchanged)
                x_frame_exchange_logits = self.local_classifier_exchange(x_frame_pool_feat_exchanged.reshape(-1, c))

            if frame_id_loss:
                return x_pool_original, x_pool_exchanged, x_frame_original, x_frame_exchanged, self.logit_scale_dense, \
                    x_track_logits, x_exchange_track_logits, \
                    x_frame_logits, x_frame_exchange_logits
            else:
                return x_pool, None, x_frame_pool, self.logit_scale_dense, self.classifier(feat_original)
        else:
            if self.inference_norm:
                x_frame_patch = None
                x_frame_pool = x_frame_pool / x_frame_pool.norm(dim=-1, keepdim=True)
            return self.l2norm(feat), x_frame_patch, x_frame_pool


