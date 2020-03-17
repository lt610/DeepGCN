import torch.nn as nn
import torch.nn.functional as F

from layers.mlp_layer import MLPLayer


class ResMLPNet(nn.Module):
    def __init__(self, num_feats, num_classes, num_hidden, num_layers, bias=False, activation=F.tanh,
                 batch_norm=False, residual=False, dropout=0):
        super(ResMLPNet, self).__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        for i in range(0, num_layers):
            if i == 0:
                self.layers.append(
                    MLPLayer(num_feats, num_hidden, bias, activation, batch_norm, residual, dropout))
            elif i == num_layers - 1:
                self.layers.append(
                    MLPLayer(num_hidden, num_classes, bias, activation, batch_norm, residual, dropout))
            else:
                self.layers.append(
                    MLPLayer(num_hidden, num_hidden, bias, activation, batch_norm, residual, dropout))

    def forward(self, g, features):
        x = None
        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(g, features)
            else:
                x = layer(g, x)
        return x