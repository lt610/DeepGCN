import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

# def gcn_message_func(edges):
#     return {'h' : edges.src['h']}
#
# def gcn_reduce_func(nodes):
#     msgs = th.sum(nodes.mailbox['h'], dim=1)
#     return {'h' : msgs}


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False, residual=False, activation=None, norm=True, dropout=0.5,
                 batch_norm=True):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.norm = norm
        self.batch_norm = batch_norm

        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feats)

        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation
        self.dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, node):
        a = 0
        t = (1 - a) * node.data['h'] + a * node.data['preh']

        h = self.linear(t)
        if self.batch_norm:
            h = self.bn(h)

        if self.activation is not None:
            h = self.activation(h)

        if self.res_fc is not None:
            h = h + self.res_fc(node.data['preh'])
        h = self.dropout(h)
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False, residual=False, activation=None):
        super(GCNLayer, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, bias, residual, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.ndata['preh'] = feature

        gcn_msg = fn.copy_src(src='h', out='m')
        gcn_reduce = fn.mean(msg='m', out='h')

        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)

        g.ndata.pop('preh')
        return g.ndata.pop('h')
