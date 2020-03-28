from layers.gcn_layer import GCNLayer
import torch.nn as nn
import torch.nn.functional as F


class ResGCNNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=False, activation=F.tanh, graph_norm=False,
                 batch_norm=False, pair_norm=False, residual=False, dropout=0, dropedge=0, init_beta=1., learn_beta=True):
        super(ResGCNNet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(0, num_layers):
            if i == 0:
                self.layers.append(
                    GCNLayer(num_feats, num_hidden, bias, activation, graph_norm, batch_norm, pair_norm, residual, dropout, dropedge, init_beta, learn_beta))
            elif i == num_layers - 1:
                self.layers.append(
                    GCNLayer(num_hidden, num_classes, bias, None, graph_norm, batch_norm, pair_norm, residual, dropout, dropedge, init_beta, learn_beta))
            else:
                self.layers.append(
                    GCNLayer(num_hidden, num_hidden, bias, activation, graph_norm, batch_norm, pair_norm, residual, dropout, dropedge, init_beta, learn_beta))

    def forward(self, g, features):
        x = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(g, features)
            else:
                x = layer(g, x)
        return x