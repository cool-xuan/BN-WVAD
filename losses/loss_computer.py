import torch
import torch.nn as nn

from .mpp_loss import MPPLoss
from .normal_loss import NormalLoss

class LossComputer(nn.Module):
    def __init__(self, w_normal=1., w_mpp=1.):
        super().__init__()
        self.w_normal = w_normal
        self.w_mpp = w_mpp
        self.mppLoss = MPPLoss()
        self.normalLoss = NormalLoss()

    def forward(self, result):
        loss = {}

        pre_normal_scores = result['pre_normal_scores']
        normal_loss = self.normalLoss(pre_normal_scores)
        loss['normal_loss'] = normal_loss

        anchors          = result['bn_results']['anchors']
        variances        = result['bn_results']['variances']
        select_normals   = result['bn_results']['select_normals']
        select_abnormals = result['bn_results']['select_abnormals']

        mpp_loss = self.mppLoss(anchors, variances, select_normals, select_abnormals)
        loss['mpp_loss'] = mpp_loss

        loss['total_loss'] = self.w_normal * normal_loss + self.w_mpp * mpp_loss

        return loss['total_loss'], loss