import torch.nn as nn
import torch.nn.functional as F
from layers.gat_layer import GATLayer


class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads):
        super(GAT, self).__init__()
        self.layer1 = GATLayer(in_dim, hidden_dim, num_heads)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = GATLayer(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, h):
        h = self.layer1(g, h)
        h = self.layer2(g, h)
        return h