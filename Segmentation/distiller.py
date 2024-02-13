import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np
from cbam import *
from da_att import *
import scipy

import math

def distillation_loss(source, target, margin):
    target = torch.max(target, margin)
    loss = torch.nn.functional.mse_loss(source, target, reduction="none")
    loss = loss * ((source > target) | (target > 0)).float()
    return loss.sum()

def build_feature_connector(t_channel, s_channel):
    C = [nn.Conv2d(s_channel, t_channel, kernel_size=1, stride=1, padding=0, bias=False),
         nn.BatchNorm2d(t_channel)]

    for m in C:
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()

    return nn.Sequential(*C)

def get_margin_from_BN(bn):
    margin = []
    std = bn.weight.data
    mean = bn.bias.data
    for (s, m) in zip(std, mean):
        s = abs(s.item())
        m = m.item()
        if norm.cdf(-m / s) > 0.001:
            margin.append(- s * math.exp(- (m / s) ** 2 / 2) / math.sqrt(2 * math.pi) / norm.cdf(-m / s) + m)
        else:
            margin.append(-3 * s)

    return torch.FloatTensor(margin).to(std.device)



class Distiller(nn.Module):
    def __init__(self, t_net, s_net, args):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        # self.cbams = nn.ModuleList([CBAM(s_channels[i], model = 'student').cuda() for i in range(len(s_channels))])
        # self.attn = PAM_Module(s_channels[3], 'student').cuda()

        self.attns = nn.ModuleList([CBAM(s_channels[i], model = 'student').cuda() for i in range(3, len(s_channels))])

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net
        self.args = args

        self.loss_divider = [8, 4, 2, 1, 1, 4*4]

    def forward(self, x, y):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)


        def overhaul():
            loss_distill = 0
            for i in range(feat_num):
                s_feats[i] = self.Connectors[i](s_feats[i])
                loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
                                / self.loss_divider[i]
            return loss_distill
        

        # Let's do it at the beginning
        for i in range(feat_num):
            s_feats[i] = self.Connectors[i](s_feats[i])
        

        kd_loss = 0

        if self.args.kd_lambda is not None: # Kd loss
          kd_loss =  self.args.kd_lambda * torch.nn.KLDivLoss()(F.log_softmax(s_out / self.temperature, dim=1), 
                                                                F.softmax(t_out / self.temperature, dim=1))
          
        
        lad_loss = 0

        if self.args.lad_lambda is not None: # LAD loss
            for i in range(3, feat_num):
                b,c,h,w = t_feats[i].shape

                (s_feats[i] / torch.norm(s_feats[i], p = 2) - t_feats[i] / torch.norm(t_feats[i], p = 2)).pow(2).sum() / (b) * self.args.lad_lambda

        pad_loss = 0

        if self.args.pad_lambda is not None: # PAD loss
            for i in range(3, feat_num):
                b,c,h,w = t_feats[i].shape

                pad_loss += (F.normalize(s_feats[i], p = 2, dim = 1) - F.normalize(t_feats[i], p = 2, dim = 1)).pow(2).sum() \
                / (h * w * b) * self.args.pad_lambda


        cad_loss = 0

        if self.args.cad_lambda is not None: # CAD loss
            for i in range(3, feat_num):
                b,c,h,w = t_feats[i].shape

                cad_loss += (F.normalize(s_feats[i], p = 2, dim = (2,3)) - F.normalize(t_feats[i], p = 2, dim = (2,3))).pow(2).sum() \
                / (c * b) * self.args.cad_lambda

        naive_loss = 0

        if self.args.naive_lambda is not None:

            for i in range(3, feat_num):
                b,c,h,w = t_feats[i].shape

                s_feats[i] = self.Connectors[i](s_feats[i])
                
                naive_loss += (s_feats[i] - t_feats[i]).pow(2).sum() / (h * w * c* b) * self.args.naive_lambda


        return s_out, kd_loss.sum() , lad_loss.sum() , pad_loss.sum() , cad_loss.sum() , naive_loss.sum()
    

    def get_cbam_modules(self):
        return self.attns