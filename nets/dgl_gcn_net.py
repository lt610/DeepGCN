import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv


class DglGCNNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=False, activation=F.tanh, graph_norm=False):
        super(DglGCNNet, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(0, num_layers):
            in_feats, out_feats = num_hidden, num_hidden
            if i == 0:
                in_feats = num_feats
            if i == num_layers - 1:
                out_feats = num_classes
            self.layers.append(
                GraphConv(in_feats, out_feats, graph_norm, bias, activation))

    def forward(self, g, features):
        x = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(g, features)
            else:
                x = layer(g, x)
        return x