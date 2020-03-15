import matplotlib.pyplot as plt
from dgl.data import citation_graph as citegrh

from utils.data import draw_part_graph, print_graph_info, cut_graph, load_data_default
from utils.save import generate_path
from dgl import DGLGraph
import numpy as np
import torch as th
import torch.nn.functional as F


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
# # g, features, labels = load_data(result)
# print(len(features))
# print(len(labels))

a1 = th.FloatTensor([[1, 2, 3],
                    [4, 5, 6]])
a2 = th.FloatTensor([[1, 1, 1],
                    [1, 1, 1]])
print(th.stack([a1, a2]))
b = th.mean(th.stack([a1, a2]), dim = 0)
print(b)

# a = 'fuck'
# print(a == 'fuck')