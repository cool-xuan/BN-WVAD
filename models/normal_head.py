import torch
import torch.nn as nn

class NormalHead(nn.Module):
    def __init__(self, in_channel=512, ratios=[16, 32], kernel_sizes=[1, 1, 1]):
        super(NormalHead, self).__init__()
        self.ratios = ratios
        self.kernel_sizes = kernel_sizes

        self.build_layers(in_channel)
        
    def build_layers(self, in_channel):
        ratio_1, ratio_2 = self.ratios
        self.conv1 = nn.Conv1d(in_channel, in_channel // ratio_1, 
                               self.kernel_sizes[0], 1, self.kernel_sizes[0] // 2)
        self.bn1 = nn.BatchNorm1d(in_channel // ratio_1)
        self.conv2 = nn.Conv1d(in_channel // ratio_1, in_channel // ratio_2, 
                               self.kernel_sizes[1], 1, self.kernel_sizes[1] // 2)
        self.bn2 = nn.BatchNorm1d(in_channel // ratio_2)
        self.conv3 = nn.Conv1d(in_channel // ratio_2, 1, 
                               self.kernel_sizes[2], 1, self.kernel_sizes[2] // 2)
        self.act = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.bns = [self.bn1, self.bn2]

    def forward(self, x):
        '''
        x: BN * C * T
        return BN * C // 64 * T and BN * 1 * T
        '''
        outputs = []
        x = self.conv1(x)
        outputs.append(x)
        x = self.conv2(self.act(self.bn1(x)))
        outputs.append(x)
        x = self.sigmoid(self.conv3(self.act(self.bn2(x))))
        outputs.append(x)
        return outputs
