from layers.gcn_layer import GCNLayer
import torch.nn as nn
import torch.nn.functional as F


class ResGCNLayerNet(nn.Module):
    def __init__(self, num_layers):
        super(ResGCNLayerNet, self).__init__()
        bias = False
        residual = True
        activation = F.tanh
        num_hidden = 112

        self.layers = nn.ModuleList()
        self.layers.append(GCNLayer(1433, num_hidden, bias, residual, activation))
        for i in range(1, num_layers - 1):
            self.layers.append(GCNLayer(num_hidden, num_hidden, bias, residual))
        self.layers.append(GCNLayer(num_hidden, 7, bias, residual))

    def forward(self, g, features):
        x = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(g, features)
            else:
                x = layer(g, x)

        return x