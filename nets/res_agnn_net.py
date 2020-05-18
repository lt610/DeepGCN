import torch.nn as nn
import torch.nn.functional as F
from layers.agnn_layer import AGNNLayer


class ResAGNNNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, project=False, bias=False, activation=F.tanh, init_beta=1., learn_beta=True,
                 batch_norm=False, residual=False, dropout=0, cutgraph=0):
        super(ResAGNNNet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        if not project:
            num_hidden = num_feats
        if num_layers == 1:
            self.layers.append(
                AGNNLayer(num_feats, num_classes, True, bias, None, init_beta, learn_beta, batch_norm,
                          residual, dropout, cutgraph))
        else:
            for i in range(0, num_layers):
                if i == 0:
                    self.layers.append(
                        AGNNLayer(num_feats, num_hidden, project, bias, activation, init_beta, learn_beta, batch_norm,
                                  residual, dropout, cutgraph))
                elif i == num_layers - 1:
                    self.layers.append(
                        AGNNLayer(num_hidden, num_classes, True, bias, None, init_beta, learn_beta, batch_norm,
                                  residual, dropout, cutgraph))
                else:
                    self.layers.append(
                        AGNNLayer(num_hidden, num_hidden, project, bias, activation, init_beta, learn_beta, batch_norm,
                                  residual, dropout, cutgraph))

    def forward(self, g, features):
        x = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(g, features)
            else:
                x = layer(g, x)
        return x