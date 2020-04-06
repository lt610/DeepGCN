import matplotlib.pyplot as plt
from dgl.data import citation_graph as citegrh
from sklearn.manifold import Isomap

from utils.data import draw_part_graph, print_graph_info, cut_graph, load_data_default, erase_features
from utils.save import generate_path
from dgl import DGLGraph
import numpy as np
import torch as th
import torch.nn.functional as F
import dgl.function as fn

# path = generate_path("result/train_result", "tmp", ".png")
# a = [i for i in range(100)]
# b = [(i * 2 +1) for i in range(100)]
# plt.plot(a, color="r")
# plt.plot(b, color="b")
# plt.savefig(path)
# plt.show()

# import torch
# print(torch.cuda.is_available())
# print(torch.cuda.device_count())
# print(torch.cuda.current_device())
# print(torch.cuda.get_device_name(0))


# G = DGLGraph()
# G.add_nodes(5)
# G.add_edges([0, 1, 2, 3, 4], [1, 2, 3, 4, 0], {'x': th.arange(15).view(5, 3)})

# print(G.nodes())
# print(G.edges())
# G.remove_edges([1, 2])
# print(G.nodes())
# print(G.edges())

# es = G.edges()
# print(es)
# len = len(es[0])
# for i in range(len):
#     print(es[0][i])
#     print(es[1][i])
#
# draw_part_graph(G.to_networkx())
# a = np.ones(10)
# b = np.where(a > 0)
# print(type(b[0]))

# result = citegrh.load_cora()
# num_feats, num_classes = result.features.shape[1], result.num_labels
# g, features, labels, train_mask, val_mask, test_mask = load_data_default(result)

# g, features, labels = load_data(result)
# print(g.in_degrees().shape)
# print(features.shape)

# a1 = th.FloatTensor([[1, 2, 3],
#                     [4, 5, 6]])
# a2 = th.FloatTensor([[1, 1, 1],
#                     [1, 1, 1]])
# print(th.stack([a1, a2]))
# b = th.mean(th.stack([a1, a2]), dim = 0)
# print(b)


# a = (1, 2) + (1,)
# b = (1, 2) + (1,) * 2

# a = th.ones(3, 4)
# b = th.tensor([1, 2, 3, 4])
# print(a + b)
# print(a * b)

# a = th.tensor([2, 1, 3, 4, 0, 0])
# values, indices = a.topk(3, largest=False, sorted=True)
# print(values)
# print(indices)
# print(a.size()[0])

# features = [[i for i in range(6)], [i for i in range(6, 12)], [i for i in range(12, 18)],
#             [i for i in range(18, 24)]]
# print(features)
# n_components = 3
# isomap = Isomap( n_components=n_components, n_neighbors=2)
# features = isomap.fit_transform(features)
# print(features)
# a = th.tensor([1, 2, 3])
# b = th.tensor([4, 5, 6])
# c = th.cat([a, b])
# print(c)
# g = DGLGraph()
# g.add_nodes(4)
# g.add_edges([0, 1, 2, 3], [1, 2, 3, 0])   # 0 -> 1, 1 -> 2
# g.ndata['y'] = th.ones(4, 4)
# g.edata
# g.apply_edges(fn.u_dot_v('y', 'y', 'cos'), edges=[0, 1])
# e = g.edata.pop('cos')
# print(e)
# t = th.where(e > 0., e, th.tensor(1.0))
# print(t)
# values, indices = e.topk(3, largest=False, sorted=False)
# print(values)
# print(indices)
# print(g.out_degrees())
# a = th.Tensor([1, 2, 3, 4, 5, 6])
# b = F.dropout(a, p=0.5)
# print(b)

