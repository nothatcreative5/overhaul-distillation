import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import norm
import numpy as np
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
    def __init__(self, t_net, s_net):
        super(Distiller, self).__init__()

        t_channels = t_net.get_channel_num()
        s_channels = s_net.get_channel_num()

        self.Connectors = nn.ModuleList([build_feature_connector(t, s) for t, s in zip(t_channels, s_channels)])

        encoder_layer = nn.TransformerEncoderLayer(d_model=s_channels[3], nhead=8, batch_first = True, dropout = 0.5)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        teacher_bns = t_net.get_bn_before_relu()
        margins = [get_margin_from_BN(bn) for bn in teacher_bns]
        for i, margin in enumerate(margins):
            self.register_buffer('margin%d' % (i+1), margin.unsqueeze(1).unsqueeze(2).unsqueeze(0).detach())

        self.t_net = t_net
        self.s_net = s_net

        self.loss_divider = [8, 4, 2, 1, 1, 4*4]

    def forward(self, x, y):

        t_feats, t_out = self.t_net.extract_feature(x)
        s_feats, s_out = self.s_net.extract_feature(x)
        feat_num = len(t_feats)

        loss_distill = 0

        'Overhaul'
        # for i in range(feat_num):
        #     s_feats[i] = self.Connectors[i](s_feats[i])
        #     loss_distill += distillation_loss(s_feats[i], t_feats[i].detach(), getattr(self, 'margin%d' % (i+1))) \
        #                     / self.loss_divider[i]


        'ICKD'

        # b, c, h, w = s_out.shape

        # s_logit = torch.reshape(s_out, (b, c, h*w))
        # t_logit = torch.reshape(t_out, (b, c, h*w)).detach()

        # # b x c x A  mul  b x A x c -> b x c x c
        # ICCT = torch.bmm(t_logit, t_logit.permute(0,2,1))
        # ICCT = torch.nn.functional.normalize(ICCT, dim = 2)

        # ICCS = torch.bmm(s_logit, s_logit.permute(0,2,1))
        # ICCS = torch.nn.functional.normalize(ICCS, dim = 2)

        # G_diff = ICCS - ICCT
        # loss_distill = (G_diff * G_diff).view(b, -1).sum() / (c)


        'Corrected ICKD'

        y_cpy = y.clone().detach()
        # y_cpy = torch.rand((b, h, w), device = 'cuda')
        y_cpy[y_cpy == 255] = 0

        b, c, h, w = s_out.shape

        s_logit = torch.reshape(s_out, (b, c, h*w))
        t_logit = torch.reshape(t_out, (b, c, h*w)).detach()

        y_cpy = torch.reshape(y_cpy, (b, h*w))

        for i in range(b):
            preds = torch.argmax(t_logit[i], dim = 0)
            indices = y_cpy[i] != preds
            val_mx = torch.max(t_logit[i]).detach()
            val_mn = torch.min(t_logit[i]).detach()

            corrected_logits = torch.ones((c, indices.sum()), device = 'cuda') * val_mn
            corrected_logits[y_cpy.long()[i][indices], torch.arange(indices.sum())] = val_mx
            t_logit[i][:, indices] = corrected_logits

        # b x c x A  mul  b x A x c -> b x c x c
        ICCT = torch.bmm(t_logit, t_logit.permute(0,2,1))
        ICCT = torch.nn.functional.normalize(ICCT, dim = 2)

        ICCS = torch.bmm(s_logit, s_logit.permute(0,2,1))
        ICCS = torch.nn.functional.normalize(ICCS, dim = 2)

        G_diff = ICCS - ICCT
        loss_distill += (G_diff * G_diff).view(b, -1).sum() / (c) * 0.1

        'SA loss'
        layer = 3
        b,c_T,h,w = t_feats[layer].shape

        M = h * w
        TF = t_feats[layer].view(b, M, c_T)

        X = torch.bmm(TF, TF.permute(0,2,1)) / np.sqrt(M)
        X = F.softmax(X, dim = 2) 

        G = torch.einsum('bji, bik -> bjk', X, TF).view(b, h, w, c_T) + TF.view(b, h, w, c_T)
        G = G.view(b, c_T, M)

        # normalize G
        G = torch.nn.functional.normalize(G, dim = 1)

        # change it for the student
        c_S = 320
    #    F_t = self.Connectors[3](self.encoder(s_feats[layer].view(b, M, c_S)).view(b, c_S, h, w))
        encoded = self.encoder(torch.reshape(s_feats[layer], (b, M, c_S)))
        F_t = self.Connectors[3](torch.reshape(encoded, (b, c_S, h, w)))

    #    F_t = F_t.view(b, c_T, M)
        F_t = torch.reshape(F_t, (b, c_T, M))

        F_t = torch.nn.functional.normalize(F_t, dim = 1)
         
        SA_loss = torch.norm(G - F_t, dim = 1)
        loss_distill += SA_loss.sum() / M * 0.2

        return s_out, loss_distill
