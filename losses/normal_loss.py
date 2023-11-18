import torch
import torch.nn as nn
import torch.nn.functional as F

class NormalLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, normal_scores):
        '''
        normal_scores: [bs, pre_k]
        '''
        loss_normal = torch.norm(normal_scores, dim=1, p=2)

        return loss_normal.mean()