import torch
import torch.nn as nn
from torch.nn.modules.module import Module
from .normal_head import NormalHead
from .translayer import Transformer

class Temporal(Module):
    def __init__(self, input_size, out_size):
        super(Temporal, self).__init__()
        self.conv_1 = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=out_size, kernel_size=3,
                    stride=1, padding=1),
            nn.ReLU(),
        )
    def forward(self, x):  
        x = x.permute(0, 2, 1)
        x = self.conv_1(x)
        x = x.permute(0, 2, 1)
        return x

class WSAD(Module):
    def __init__(self, input_size, flag, args):
        super().__init__()
        self.flag = flag
        self.args = args
        
        self.ratio_sample = args.ratio_sample
        self.ratio_batch = args.ratio_batch
        
        self.ratios = args.ratios
        self.kernel_sizes = args.kernel_sizes

        self.normal_head = NormalHead(in_channel=512, ratios=args.ratios, kernel_sizes=args.kernel_sizes)
        self.embedding = Temporal(input_size,512)
        self.selfatt = Transformer(512, 2, 4, 128, 512, dropout = 0)
        self.step = 0
    
    def get_normal_scores(self, x, ncrops=None):
        new_x  = x.permute(0, 2, 1)
        
        outputs = self.normal_head(new_x)
        normal_scores = outputs[-1]
        xhs = outputs[:-1]
        
        if ncrops:
            b = normal_scores.shape[0] // ncrops
            normal_scores = normal_scores.view(b, ncrops, -1).mean(1)
        
        return xhs, normal_scores
    
    def get_mahalanobis_distance(self, feats, anchor, var, ncrops = None):
        distance = torch.sqrt(torch.sum((feats - anchor[None, :, None]) ** 2 / var[None, :, None], dim=1))
        if ncrops:
            bs = distance.shape[0] // ncrops
            # b x t
            distance = distance.view(bs, ncrops, -1).mean(1)
        return distance
    
    def pos_neg_select(self, feats, distance, ncrops):
        batch_select_ratio = self.ratio_batch
        sample_select_ratio = self.ratio_sample
        bs, c, t = feats.shape
        select_num_sample = int(t * sample_select_ratio)
        select_num_batch = int(bs // 2 * t * batch_select_ratio)
        feats = feats.view(bs, ncrops, c, t).mean(1) # b x c x t
        nor_distance = distance[:bs // 2] # b x t
        nor_feats = feats[:bs // 2].permute(0, 2, 1) # b x t x c
        abn_distance = distance[bs // 2:] # b x t
        abn_feats = feats[bs // 2:].permute(0, 2, 1) # b x t x c
        abn_distance_flatten = abn_distance.reshape(-1)
        abn_feats_flatten = abn_feats.reshape(-1, c)
        
        mask_select_abnormal_sample = torch.zeros_like(abn_distance, dtype=torch.bool)
        topk_abnormal_sample = torch.topk(abn_distance, select_num_sample, dim=-1)[1]
        mask_select_abnormal_sample.scatter_(1, topk_abnormal_sample, True)
        
        mask_select_abnormal_batch = torch.zeros_like(abn_distance_flatten, dtype=torch.bool)
        topk_abnormal_batch = torch.topk(abn_distance_flatten, select_num_batch, dim=-1)[1]
        mask_select_abnormal_batch.scatter_(0, topk_abnormal_batch, True)
        
        mask_select_abnormal = mask_select_abnormal_batch | mask_select_abnormal_sample.reshape(-1)
        select_abn_feats = abn_feats_flatten[mask_select_abnormal]
        
        num_select_abnormal = torch.sum(mask_select_abnormal)
        
        k_nor = int(num_select_abnormal / (bs // 2)) + 1
        topk_normal_sample = torch.topk(nor_distance, k_nor, dim=-1)[1]
        select_nor_feats = torch.gather(nor_feats, 1, topk_normal_sample[..., None].expand(-1, -1, c))
        select_nor_feats = select_nor_feats.permute(1, 0, 2).reshape(-1, c)
        select_nor_feats = select_nor_feats[:num_select_abnormal]
        
        return select_nor_feats, select_abn_feats

    def forward(self, x):
        if len(x.size()) == 4:
            b, n, t, d = x.size()
            x = x.reshape(b * n, t, d)
        else:
            b, t, d = x.size()
            n = 1
        x = self.embedding(x)
        x = self.selfatt(x)
        
        normal_feats, normal_scores = self.get_normal_scores(x, n)
        
        anchors = [bn.running_mean for bn in self.normal_head.bns]
        variances = [bn.running_var for bn in self.normal_head.bns]

        distances = [self.get_mahalanobis_distance(normal_feat, anchor, var, ncrops=n) for normal_feat, anchor, var in zip(normal_feats, anchors, variances)]

        if self.flag == "Train":
            
            select_normals = []
            select_abnormals = []
            for feat, distance in zip(normal_feats, distances):
                select_feat_normal, select_feat_abnormal = self.pos_neg_select(feat, distance, n)
                select_normals.append(select_feat_normal[..., None])
                select_abnormals.append(select_feat_abnormal[..., None])

            bn_resutls = dict(
                anchors = anchors,
                variances = variances,
                select_normals = select_normals,
                select_abnormals = select_abnormals, 
            )

            return {
                    'pre_normal_scores': normal_scores[0:b // 2],
                    'bn_results': bn_resutls,
                }
        else:

            distance_sum = sum(distances)

            return distance_sum * normal_scores