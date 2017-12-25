import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from network.prior import PriorBox
from network.detect import Detect


# module1: conv+bn+leaky_relu
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, relu=True, same_padding=False):
        super(ConvLayer, self).__init__()
        padding = kernel_size // 2 if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.1, inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


# reorg layer
class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        B, C, H, W = x.size()
        s = self.stride
        x = x.view(B, C, H // s, s, W // s, s).transpose(3, 4).contiguous()
        x = x.view(B, C, H // s * W // s, s * s).transpose(2, 3).contiguous()
        x = x.view(B, C, s * s, H // s, W // s).transpose(1, 2).contiguous()
        return x.view(B, s * s * C, H // s, W // s)


# darknet feature detector
class DarknetBone(nn.Module):
    def __init__(self, cfg):
        super(DarknetBone, self).__init__()
        self.cfg = cfg
        in_channel, out_channel = 3, 16 if cfg.tiny else 32
        flag, pool, size_flag = cfg.flag, cfg.pool, cfg.size_flag
        layers1, layers2 = [], []
        for i in range(cfg.num):
            ksize = 1 if i in size_flag else 3
            if i < 13:
                layers1.append(ConvLayer(in_channel, out_channel, ksize, same_padding=True))
                layers1.append(nn.MaxPool2d(2)) if i in pool else None
                layers1 += [nn.ReflectionPad2d([0, 1, 0, 1]), nn.MaxPool2d(2, 1)] if i == 5 and cfg.tiny else []
            else:
                layers2.append(nn.MaxPool2d(2)) if i in pool else None
                layers2.append(ConvLayer(in_channel, out_channel, ksize, same_padding=True))
            in_channel, out_channel = out_channel, out_channel * 2 if flag[i] else out_channel // 2
        self.main1 = nn.Sequential(*layers1)
        self.main2 = nn.Sequential(*layers2)

    def forward(self, x):
        xd = self.main1(x)
        if self.cfg.tiny:
            return xd
        else:
            x = self.main2(xd)
            return x, xd


# YOLO
class Yolo(nn.Module):
    def __init__(self, cfg):
        super(Yolo, self).__init__()
        self.cfg = cfg
        self.prior = Variable(PriorBox(cfg).forward(), volatile=True)
        self.darknet = DarknetBone(cfg)
        if cfg.tiny:
            out = 1024 if cfg.voc else 512
            self.conv = nn.Sequential(
                ConvLayer(1024, out, 3, same_padding=True),
                nn.Conv2d(out, cfg.anchor_num * (cfg.class_num + 5), 1))
        else:
            self.conv1 = nn.Sequential(
                ConvLayer(1024, 1024, 3, same_padding=True),
                ConvLayer(1024, 1024, 3, same_padding=True))
            self.conv2 = nn.Sequential(
                ConvLayer(512, 64, 1, same_padding=True),
                ReorgLayer(2))
            self.conv = nn.Sequential(
                ConvLayer(1280, 1024, 3, same_padding=True),
                nn.Conv2d(1024, cfg.anchor_num * (cfg.class_num + 5), 1))

    def forward(self, x, img_shape=None):
        if self.cfg.tiny:
            x = self.conv(self.darknet(x))
        else:
            x1, x2 = self.darknet(x)
            x = self.conv(torch.cat([self.conv2(x2), self.conv1(x1)], 1))
        # extract each part
        b, c, h, w = x.size()
        feat = x.permute(0, 2, 3, 1).contiguous().view(b, -1, self.cfg.anchor_num, self.cfg.class_num + 5)
        box_xy, box_wh = F.sigmoid(feat[..., 0:2]), feat[..., 2:4].exp()
        box_conf, score_pred = F.sigmoid(feat[..., 4:5]), feat[..., 5:].contiguous()
        box_prob = F.softmax(score_pred, dim=3)
        box_pred = torch.cat([box_xy, box_wh], 3)
        # TODO: add training phase
        if self.training:
            return x
        else:
            width, height = img_shape
            img_shape = Variable(torch.Tensor([[width, height, width, height]]))
            if self.cfg.cuda: self.prior, img_shape = self.prior.cuda(), img_shape.cuda()
            self.prior = self.prior.view_as(box_pred)
            return Detect(self.cfg, self.cfg.eval)(box_pred, box_conf, box_prob, self.prior, img_shape)


# interface --- construct different type yolo model
def yolo(cfg):
    model = Yolo(cfg)
    return model


# weight initialize
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, (m.kernel_size[0] ** 2 * m.out_channels) ** 0.5)
    elif classname.find('BatchNorm') != -1:
        # Estimated variance, must be around 1
        m.weight.data.normal_(1.0, 0.02)
        # Estimated mean, must be around 0
        m.bias.data.fill_(0)
