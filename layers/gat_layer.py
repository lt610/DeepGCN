import torch as th
import torch.nn as nn
import torch.nn.functional as F

from layers.gcn_layer import cal_gain


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SingleHeadGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, activation=F.elu, graph_norm=False,
                 batch_norm=False, residual=False, dropout=0):
        super(SingleHeadGATLayer, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.attn_fc = nn.Linear(2 * out_dim, 1, bias=False)
        self.activation = activation
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain("leaky_relu")
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=cal_gain(self.activation))

    def edge_attention(self, edges):
        z2 = th.cat([edges.src['z'], edges.dst['z']], dim=1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = th.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, g, features):
        g = g.local_var()
        h_pre = features

        if self.graph_norm:
            norm = th.pow(g.in_degrees().float().clamp(min=1), -1)
            shp = norm.shape + (1,) * (features.dim() - 1)
            norm = th.reshape(norm, shp).to(features.device)
            features = features * norm

        z = self.fc(features)
        g.ndata['z'] = z
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        h = g.ndata['h']

        if self.batch_norm:
            h = self.bn(h)
        if self.activation:
            h = self.activation(h)
        if self.residual:
            h = h + self.res_fc(h_pre)
        h = self.dropout(h)
        return h


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', activation=F.elu, graph_norm=False,
                 batch_norm=False, residual=False, dropout=0):
        super(GATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(SingleHeadGATLayer(in_dim, out_dim, activation, graph_norm, batch_norm, residual, dropout))
        self.merge = merge

    def forward(self, g, features):
        head_outs = [attn_head(g, features) for attn_head in self.heads]
        if self.merge == 'cat':
            return th.cat(head_outs, dim=1)
        else:
            return th.mean(th.stack(head_outs), dim=0)
