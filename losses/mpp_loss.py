import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

class MPPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.w_triplet = [5, 20]

    def forward(self, anchors, variances, select_normals, select_abnormals):
        losses_triplet = []

        def mahalanobis_distance(mu, x, var):
            return torch.sqrt(torch.sum((x - mu)**2 / var, dim=-1))
        
        for anchor, var, pos, neg, wt in zip(anchors, variances, select_normals, select_abnormals, self.w_triplet):
            triplet_loss = nn.TripletMarginWithDistanceLoss(margin=1, distance_function=partial(mahalanobis_distance, var=var))
            
            B, C, k = pos.shape
            pos = pos.permute(0, 2, 1).reshape(B*k, -1)
            neg = neg.permute(0, 2, 1).reshape(B*k, -1)
            loss_triplet = triplet_loss(anchor[None, ...].repeat(B*k, 1), pos, neg)
            losses_triplet.append(loss_triplet * wt)
        
        return sum(losses_triplet)