import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import APPNPConv


class DglAPNNNet(nn.Module):
    def __init__(self, num_feats, num_classes, k, alpha, bias=False, activation=None):
        super(DglAPNNNet, self).__init__()
        self.layer = APPNPConv(k, alpha)
        self.linear = nn.Linear(num_feats, num_classes, bias)
        self.activation = activation

    def forward(self, g, features):

        x = self.layer(g, features)
        x = self.linear(x)
        if self.activation:
            x = self.activation(x)
        return x
