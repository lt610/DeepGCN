import torch.nn as nn
import torch.nn.functional as F
from layers.gat_layer import GATLayer


class ResGATNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, num_heads, merge='cat', activation=F.elu,
                 graph_norm=False,
                 batch_norm=False, residual=False, dropout=0):
        super(ResGATNet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.layers.append(GATLayer(num_feats, num_hidden, num_heads, merge, activation,
                                    graph_norm, batch_norm, residual, dropout))
        for i in range(1, num_layers - 1):
            self.layers.append(GATLayer(num_hidden * num_heads, num_hidden, num_heads, merge, activation,
                                        graph_norm, batch_norm, residual, dropout))
        self.layers.append(
            GATLayer(num_hidden * num_heads, num_classes, 1, 'mean', activation,
                                        graph_norm, batch_norm, residual, dropout))

    def forward(self, g, features):
        x = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(g, features)
            else:
                x = layer(g, x)
        return x
