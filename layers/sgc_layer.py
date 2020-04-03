import torch as th
from torch import nn
import dgl.function as fn
from torch.nn import functional as F
import numpy as np
from layers.pair_norm import PairNorm


class SGCLayer(nn.Module):
    def __init__(self, in_feats, out_feats, k=1, cached=False, bias=False, graph_norm=False, pair_norm=False, dropedge=0, cutgraph=0):
        super(SGCLayer, self).__init__()
        self.fc = nn.Linear(in_feats, out_feats, bias=bias)
        self._cached = cached
        self._cached_h = None
        self._k = k
        self.graph_norm = graph_norm
        self.pair_norm = pair_norm
        if pair_norm:
            self.pn = PairNorm(mode='PN-SCS', scale=1)
        self.edge_drop = nn.Dropout(dropedge)
        self.graph_cut = cutgraph
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(self, g, features):
        g = g.local_var()
        # if self.graph_cut > 0:
        #     k1 = int(g.number_of_nodes() * 0.5)
        #     degrees = g.in_degrees() + g.out_degrees()
        #     _, indices = degrees.topk(k1, largest=False, sorted=False)
        #     edges = th.cat([g.in_edges(indices, 'eid'), g.out_edges(indices, 'eid')])

        if self._cached_h is not None:
            features = self._cached_h
        else:
            # compute normalization
            if self.graph_norm:
                degs = g.in_degrees().float().clamp(min=1)
                norm = th.pow(degs, -0.5)
                norm = norm.to(features.device).unsqueeze(1)

            # compute (D^-1 A^k D)^k X
            for i in range(self._k):
                w = th.ones(g.number_of_edges(), 1).to(features.device)
                if i > -1:
                    w = self.edge_drop(w)
                    if self.graph_cut > 0:
                        # g.ndata['norm_h'] = F.normalize(features, p=2, dim=-1)
                        # g.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'), edges=edges)
                        # t = g.edata.pop('cos')
                        # e = th.where(t == 0, t, th.tensor(1.0).to(features.device))
                        #
                        # k = int(edges.size()[0] * self.graph_cut)
                        # _, indices = e.topk(k, largest=False, sorted=False)

                        g.ndata['norm_h'] = F.normalize(features, p=2, dim=-1)
                        g.apply_edges(fn.u_dot_v('norm_h', 'norm_h', 'cos'))
                        e = g.edata.pop('cos')
                        k = int(e.size()[0] * self.graph_cut)
                        _, indices = e.topk(k, largest=False, sorted=False)

                        w[indices] = 0
                g.edata['w'] = w
                if self.graph_norm:
                    features = features * norm
                if self.pair_norm:
                    self.pn(features)
                g.ndata['h'] = features
                g.update_all(fn.u_mul_e('h', 'w', 'm'),
                             fn.sum('m', 'h'))
                features = g.ndata.pop('h')
                if self.graph_norm:
                    features = features * norm
                if self.pair_norm:
                    self.pn(features)

            # cache feature
            if self._cached:
                self._cached_h = features
        return self.fc(features)
