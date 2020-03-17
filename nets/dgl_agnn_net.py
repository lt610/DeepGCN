import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import AGNNConv


class DglAPPNPNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_layers, bias=False):
        super(DglAPPNPNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(0, num_layers):
            self.layers.append(AGNNConv())
        self.linear = nn.Linear(num_feats, num_classes, bias=bias)

    def forward(self, g, features):
        x = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(g, features)
            else:
                x = layer(g, x)
        x = self.linear(x)
        return x