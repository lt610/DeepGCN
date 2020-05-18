import torch as th
from dgl.nn.pytorch import edge_softmax
from torch import nn
from torch.nn import functional as F
import dgl.function as fn

from layers.gcn_layer import cal_gain


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class AGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, project=True, bias=False, activation=None, init_beta=1., learn_beta=True,
                 batch_norm=False, residual=False, dropout=0, cutgraph=0):
        super(AGNNLayer, self).__init__()
        if learn_beta:
            self.beta = nn.Parameter(th.Tensor([init_beta]))
        else:
            self.register_buffer('beta', th.Tensor([init_beta]))
        self.project = project
        if project:
            self.linear = nn.Linear(in_dim, out_dim, bias)
        self.activation = activation
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_dim)
        self.residual = residual
        if residual:
            if in_dim != out_dim:
                self.res_fc = nn.Linear(in_dim, out_dim, bias)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.dropout = nn.Dropout(dropout)
        self.graph_cut = cutgraph
        self.reset_parameters()

    def reset_parameters(self):
        gain = cal_gain(self.activation)
        if self.project:
            nn.init.xavier_normal_(self.linear.weight, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, g, features):
        h_pre = features
        g = g.local_var()
        g.ndata['h'] = features

        g.ndata['norm_h'] = F.normalize(features, p=2, dim=-1)
        g.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'))
        cos = g.edata.pop('cos')
        e = self.beta * cos
        if self.graph_cut > 0:
            k = int(e.size()[0] * self.graph_cut)
            _, indices = e.topk(k, largest=False, sorted=False)
            e[indices] = 0

        g.edata['p'] = edge_softmax(g, e)

        g.update_all(fn.u_mul_e('h', 'p', 'm'), fn.sum('m', 'h'))
        h = g.ndata['h']
        if self.project:
            h = self.linear(h)
        if self.activation:
            h = self.activation(h)
        if self.residual:
            h = h + self.res_fc(h_pre)
        h = self.dropout(h)
        return h
