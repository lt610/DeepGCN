import torch.nn as nn
import torch.nn.functional as F
import torch as th

from layers.sgc_layer import SGCLayer


class RootSGCNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, k=1, cached=False, bias=False, dropedge=0, cutgraph=0):
        super(RootSGCNet, self).__init__()
        self.sgc = SGCLayer(num_feats, num_hidden, k, cached, bias, dropedge, cutgraph)
        self.linear = nn.Linear(num_feats, num_hidden, bias)
        self.project = nn.Linear(num_hidden * 2, num_classes, bias)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.xavier_normal_(self.project.weight)

    def forward(self, g, features):
        x1 = self.linear(features)
        x2 = self.sgc(g, features)
        x3 = self.project(th.cat([x1, x2], 1))
        return x3
