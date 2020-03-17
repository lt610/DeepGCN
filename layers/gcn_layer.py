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
from layers.pair_norm import PairNorm

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


def cal_gain(fun, param=None):
    gain = 1
    if fun is F.sigmoid:
        gain = nn.init.calculate_gain('sigmoid')
    if fun is F.tanh:
        gain = nn.init.calculate_gain('tanh')
    if fun is F.relu:
        gain = nn.init.calculate_gain('relu')
    if fun is F.leaky_relu:
        gain = nn.init.calculate_gain('leaky_relu', param)
    return gain


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NodeApplyModule(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, node):
        h = self.linear(node.data['h'])
        return {'h': h}


class GCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False, activation=None, graph_norm=False,
                 batch_norm=False, pair_norm=False, residual=False, dropout=0, init_beta=1., learn_beta=True):
        super(GCNLayer, self).__init__()
        self.apply_mod = NodeApplyModule(in_dim, out_dim, bias)
        self.activation = activation
        self.graph_norm = graph_norm
        self.batch_norm = batch_norm
        self.pair_norm = pair_norm
        self.residual = residual
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        if pair_norm:
            self.pn = PairNorm(mode='PN-SCS', scale=1)
        if residual:
            if learn_beta:
                self.beta = nn.Parameter(th.Tensor([init_beta]))
            else:
                self.register_buffer('beta', th.Tensor([init_beta]))
            # self.alpha = nn.Parameter(th.Tensor([1.]))
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        nn.init.xavier_normal_(self.apply_mod.linear.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, g, features):
        g = g.local_var()
        h_pre = features
        if self.graph_norm:
            degs = g.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(features.device).unsqueeze(1)
            features = features * norm

        g.ndata['h'] = features
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        h = g.ndata['h']

        if self.batch_norm:
            h = self.bn(h)
        if self.pair_norm:
            h = self.pn(h)
        if self.activation is not None:
            h = self.activation(h)
        if self.res_fc is not None:
            # # h = h + self.res_fc(h_pre)
            # print("beta:{}".format(self.beta))
            h = h + self.beta * self.res_fc(h_pre)
            # h = self.alpha * h + self.beta * self.res_fc(h_pre)
            # h = (1 - self.beta) * h + self.beta * self.res_fc(h_pre)
        h = self.dropout(h)
        return h

