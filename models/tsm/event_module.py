import torch
import torch.nn as nn
from resnet import resnet50
from torch.nn import init
from tsm_resnet import tsmresnet50

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


class EventIntegralModule(nn.Module):
    def __init__(self, arch='resnet50', seq_len=6):
        super(EventIntegralModule, self).__init__()
        self.seq_len = seq_len
        if 'resnet50' in arch:
            resnet_model = tsmresnet50(num_segments=seq_len, pretrained=True)
            resnet_model1 = tsmresnet50(num_segments=seq_len, pretrained=True)
        else:
            raise NotImplementedError

        self.conv1 = list(resnet_model.children())[0]
        self.bn1 = list(resnet_model.children())[1]
        self.relu = nn.ReLU(inplace=True)

        self.layer1_bak = nn.Sequential(*list(resnet_model.children())[4])

        self.event_conv = nn.Conv2d(self.conv1.out_channels*seq_len, self.conv1.out_channels, kernel_size=1, stride=1, padding=0)
        self.event_bn = nn.BatchNorm2d(3)
        self.event_relu = nn.ReLU(inplace=True)
        self.event_conv2 = nn.Sequential(*list(resnet_model1.children())[4])

        window = torch.zeros(seq_len, self.conv1.out_channels*seq_len)
        for i in range(seq_len):
            window[i, :(i+1)*self.conv1.out_channels] = 1.
        self.register_buffer('window', window)
        self.alpha = nn.Parameter(torch.zeros(1))

        self.event_conv.apply(weights_init_kaiming)
        self.event_bn.apply(weights_init_kaiming)

    def forward(self, x):
        # x.shape = (B,T,C,H,W)
        B, T, C, H, W = x.shape
        assert T == self.seq_len, "seq_len = {} in data doesn't equal to {} in model".format(T, self.seq_len)
        x = x.reshape([B * T, C, H, W])
        print(f'x1 shape: {x.shape}')
        # resnet conv1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x) #(B*T, 64, H, W)
        print(f'x2 shape: {x.shape}')
        # integral module
        int_input = x.reshape(B, -1, H, W) #(B, T*64, H, W)
        print(f'int input shape1: {int_input.shape}')
        int_input = torch.permute(int_input, (0, 2, 3, 1))  #(B, H, W, T*64)
        print(f'int input shape2: {int_input.shape}')
        int_input = int_input.unsqueeze(3) # (B, H, W, 1, T*64)
        print(f'int input shape3: {int_input.shape}')
        integral_feat = torch.mul(int_input, self.window) #(B, H, W, T, T*64)
        print(f'integral  shape1: {integral_feat.shape}')
        integral_feat = torch.permute(integral_feat, (0, 3, 4, 1, 2)) #(B, T, T*64, H, W)
        print(f'integral  shape2: {integral_feat.shape}')
        integral_feat = integral_feat.reshape(-1, integral_feat.shape[2], integral_feat.shape[3], integral_feat.shape[4]) #(B*T, T*64, H, W)
        print(f'integral  shape3: {integral_feat.shape}')
        integral_feat = self.event_relu(self.event_bn(self.event_conv(integral_feat))) #(B*T, conv1_out_dim, H, W)
        print(f'integral  shape4: {integral_feat.shape}')
        integral_feat = self.event_conv2(integral_feat) # (#(B*T, layer1_hid_dim, H, W))
        print(f'integral  shape5: {integral_feat.shape}')
        integral_feat = integral_feat.reshape(B, -1, integral_feat.shape[1], integral_feat.shape[2], integral_feat[3]) #(B, T, convlayer1_hid_dim, H, W)
        print(f'integral  shape6: {integral_feat.shape}')

        # merge the two parts
        x = self.layer1_bak(x)
        print(f'layer1 bak shape1: {x.shape}')
        x = x.reshape(B, -1, x.shape[1], x.shapep[2], x.shape[3]) #(B, T, conv2_hid_dim, H, W)
        print(f'layer1 bak shape2: {x.shape}')
        x = x + self.alpha * integral_feat
        exit()
        return x


if __name__ == '__main__':
    resnet_model = tsmresnet50(num_segments=6, pretrained=True)
    conv1 = list(resnet_model.children())[0]
    print(conv1.out_channels)
    layers_bak = nn.Sequential(*list(resnet_model.children())[4])
    print(layers_bak[2].conv3.out_channels)

