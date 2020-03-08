from layers.gcn_layer import GCNLayer
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class DenseGCNNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=False, activation=F.tanh, graph_norm=False,
                 batch_norm=False, dropout=0):
        super(DenseGCNNet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(num_feats, num_hidden, bias, activation, graph_norm, batch_norm, dropout))
        for i in range(1, num_layers - 1):
            self.layers.append(GCNLayer(num_hidden * i, num_hidden, bias, activation, graph_norm, batch_norm, dropout))
        self.layers.append(GCNLayer(num_hidden * (num_layers - 1), num_classes, bias, activation, graph_norm, batch_norm, dropout))

    def forward(self, g, features):
        pres = []
        x = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(g, features)
                pres.append(x)
            elif i == self.num_layers-1:
                x = layer(g, th.cat(pres, 1))
            else:
                x = layer(g, th.cat(pres, 1))
                pres.append(x)
        return x