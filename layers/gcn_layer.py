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

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.mean(msg='m', out='h')


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_normal_(self.linear.weight, gain=gain)

    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, bias=False, activation=None, graph_norm=False,
                 batch_norm=False, residual=False, dropout=0):
        super(GCNLayer, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, bias)
        self.activation = activation
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.residual = residual
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_feats)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, out_feats, bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('tanh')
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, g, feature):
        h_pre = feature
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        h = g.ndata['h']
        if self.graph_norm:
            print("not")
        if self.batch_norm:
            h = self.bn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.res_fc is not None:
            h = h + self.res_fc(h_pre)
        h = self.dropout(h)
        return h

