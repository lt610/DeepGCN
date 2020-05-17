import itertools

import matplotlib.pyplot as plt
from dgl.data import citation_graph as citegrh
from sklearn.manifold import Isomap
from sklearn.grid_search import GridSearchCV
from utils.data import draw_part_graph, print_graph_info, cut_graph, load_data_default, erase_features, draw
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

# a = 1
# b = 2
# print(id(a))
# print(id(b))

# a = [1, 2, ]
# b = ['a', 'b']
# c = [True, False]
# d = itertools.product(a, b, c)
# for i in d:
#     print(type(i))
#     print(i)
#     print(i[0])
#
# # a, b, c = 1, 2,\
# #           3
# # print(c)
# a = {}
# # a['a'] = 1
# # print(a)
#
# a = [1, 2, 3]
# print(np.mean(a))
# ls = [3, 4, 5, 8, 9]
# with open('result/train_result/result.txt', 'a') as f:
#     f.write(str(ls)+'\n')
# x = np.arange(2, 11)
# y1 = x
# y2 = x * 2
# y3 = x * 3
# y4 = x * 4
#
# plt.title('Pubmed')
# plt.xlabel('Number of Layers')
# plt.ylabel('Accuracy')
# plt.xticks(x)
# plt.xlim(1, 10)
# # plt.ylim(0, 1)
# plt.plot(x, y1, color='green', linestyle='-', marker='^', label='train')
# plt.plot(x, y2, color='green', linestyle=':', marker='^', label='train1')
# plt.plot(x, y3, color='blue', linestyle='-', marker='o', label='test')
# plt.plot(x, y4, color='blue', linestyle=':', marker='o', label='test1')
# plt.legend()
# plt.show()
#
# x = np.arange(1, 17)
# y1 = x
# y2 = x * 2
# y3 = x * 3
# y4 = x * 4
#
# draw("Cora", "WD", y1, y2, y3, y4)

# title = 'Pubmed'
# method = 'WD'
# train_accs0 = [64.94, 90.81, 88.81, 87.84, 87.3, 87.09, 86.54, 86.24, 85.55, 85.61, 86.23, 85.63, 85.58, 85.03, 56.97, 48.12]
# train_accs1 = [76.86, 86.72, 87.67, 87.17, 86.48, 86.17, 85.95, 85.53, 84.13, 85.56, 86.23, 85.63, 84.79, 85.13, 72.93, 76.77]
# test_accs0 = [63.39, 86.76, 85.45, 85.45, 84.74, 84.74, 84.69, 84.53, 84.63, 84.84, 84.84, 84.13, 84.43, 83.67, 56.9, 47.77]
# test_accs1 = [75.35, 86.61, 86.21, 85.34, 85.24, 84.79, 84.53, 84.43, 83.72, 84.64, 84.84, 84.13, 84.51, 83.74, 72.46, 76.22]
# draw(title, method, train_accs0, train_accs1, test_accs0, test_accs1)

# title = 'Pubmed'
# method = 'DO'
# train_accs0 = [64.94, 90.81, 88.81, 87.84, 87.3, 87.09, 86.54, 86.24, 85.55, 85.61, 86.23, 85.63, 85.58, 85.03, 56.97, 48.12]
# train_accs1 = [68.4, 89.79, 87.53, 87.26, 86.45, 86.05, 86.14, 85.47, 84.36, 82.83, 83.68, 72.09, 84.72, 79.59, 84.1, 83.12]
# test_accs0 = [63.39, 86.76, 85.45, 85.45, 84.74, 84.74, 84.69, 84.53, 84.63, 84.84, 84.84, 84.13, 84.43, 83.67, 56.9, 47.77]
# test_accs1 = [65.97, 87.22, 85.5, 85.45, 84.74, 84.18, 84.33, 84.23, 83.16, 81.8, 82.56, 70.65, 82.51, 78.77, 82.91, 82.25]
# draw(title, method, train_accs0, train_accs1, test_accs0, test_accs1)

# title = 'Pubmed'
# method = 'ES'
# train_accs1 = [64.94, 90.81, 88.81, 87.84, 87.3, 87.09, 86.54, 86.24, 85.55, 85.61, 86.23, 85.63, 85.58, 85.03, 56.97, 48.12]
# train_accs0 = [64.84, 90.8, 89.81, 89.69, 88.79, 87.47, 86.77, 87.18, 86.07, 86.61, 85.68, 84.88, 58.61, 41.48, 41.9, 41.64]
# test_accs1 = [63.39, 86.76, 85.45, 85.45, 84.74, 84.74, 84.69, 84.53, 84.63, 84.84, 84.84, 84.13, 84.43, 83.67, 56.9, 47.77]
# test_accs0 = [63.13, 86.61, 85.55, 85.55, 85.5, 84.33, 84.03, 84.18, 84.23, 83.77, 83.98, 83.77, 58.37, 41.73, 42.85, 42.24]
# draw(title, method, train_accs0, train_accs1, test_accs0, test_accs1)

# title = 'Pubmed'
# method = 'Xa'
# train_accs1 = [64.94, 90.81, 88.34, 87.65, 86.97, 86.27, 86.18, 86.11, 86.04, 85.34, 85.69, 85.68, 60.88, 58.71, 47.62, 57.0]
# train_accs0 = [61.51, 90.14, 88.31, 88.69, 87.52, 86.08, 84.96, 86.02, 69.82, 85.6, 61.28, 70.46, 48.76, 70.19, 55.16, 48.12]
# test_accs1 = [63.39, 86.61, 85.34, 85.24, 85.19, 83.92, 84.33, 84.69, 84.53, 84.18, 84.33, 84.23, 61.21, 58.87, 47.72, 57.3]
# test_accs0 = [60.04, 86.31, 85.24, 85.75, 84.74, 83.77, 84.13, 83.82, 70.28, 83.42, 60.8, 70.44, 49.95, 70.64, 53.96, 47.77]
# draw(title, method, train_accs0, train_accs1, test_accs0, test_accs1)

# title = 'Pubmed'
# method = 'GC'
# train_accs0 = [64.94, 90.81, 88.34, 87.65, 86.97, 86.27, 86.18, 86.11, 86.04, 85.34, 85.69, 85.68, 60.88, 58.71, 47.62, 57.0]
# train_accs1 = [64.94, 90.81, 88.81, 87.84, 87.3, 87.09, 86.54, 86.67, 86.23, 85.91, 83.82, 84.97, 84.38, 85.39, 83.62, 60.53]
# test_accs0 = [63.39, 86.61, 85.34, 85.24, 85.19, 83.92, 84.33, 84.69, 84.53, 84.18, 84.33, 84.23, 61.21, 58.87, 47.72, 57.3]
# test_accs1 = [63.39, 86.76, 85.45, 85.45, 84.74, 84.74, 84.74, 84.58, 84.63, 83.92, 84.28, 84.43, 83.82, 84.23, 83.32, 60.9]
# draw(title, method, train_accs0, train_accs1, test_accs0, test_accs1)

# title = 'Pubmed'
# method = 'BN'
# train_accs0 = [64.94, 90.81, 88.34, 87.65, 86.97, 86.27, 86.18, 86.11, 86.04, 85.34, 85.69, 85.68, 60.88, 58.71, 47.62, 57.0]
# train_accs1 = [87.59, 94.71, 95.82, 93.16, 94.29, 93.21, 90.68, 89.65, 87.28, 87.57, 87.0, 87.78, 86.57, 86.88, 85.95, 86.17]
# test_accs0 = [63.39, 86.61, 85.34, 85.24, 85.19, 83.92, 84.33, 84.69, 84.53, 84.18, 84.33, 84.23, 61.21, 58.87, 47.72, 57.3]
# test_accs1 = [85.6, 87.93, 86.41, 86.46, 85.4, 84.94, 84.63, 84.99, 83.72, 84.08, 83.92, 83.72, 83.47, 84.08, 83.92, 83.42]
# draw(title, method, train_accs0, train_accs1, test_accs0, test_accs1)

title = 'Pubmed'
method = 'BN'
train_accs0 = [64.94, 90.81, 88.34, 87.65, 86.97, 86.27, 86.18, 86.11, 86.04, 85.34, 85.69, 85.68, 60.88, 58.71, 47.62, 57.0]
train_accs1 = [87.59, 94.71, 95.82, 93.16, 94.29, 93.21, 90.68, 89.65, 87.28, 87.57, 87.0, 87.78, 86.57, 86.88, 85.95, 86.17]
test_accs0 = [63.39, 86.61, 85.34, 85.24, 85.19, 83.92, 84.33, 84.69, 84.53, 84.18, 84.33, 84.23, 61.21, 58.87, 47.72, 57.3]
test_accs1 = [85.6, 87.93, 86.41, 86.46, 85.4, 84.94, 84.63, 84.99, 83.72, 84.08, 83.92, 83.72, 83.47, 84.08, 83.92, 83.42]
draw(title, method, train_accs0, train_accs1, test_accs0, test_accs1)