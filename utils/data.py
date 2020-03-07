import numpy as np
import networkx as nx
import torch as th
from dgl import DGLGraph
import torch as np
import random


def load_data_default(data):
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.BoolTensor(data.train_mask)
    val_mask = th.BoolTensor(data.val_mask)
    test_mask = th.BoolTensor(data.test_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, val_mask, test_mask


def load_data(data):
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels


def split_data(data, train_ratio=0.6, val_ratio=0.2, random_seed=None):
    if random_seed:
        np.random.seed(random_seed)
    shuffled_indices = np.random.permutation(len(data))
    train_set_size = int(len(data) * train_ratio)
    val_set_size = int(len(data) * val_ratio)
    train_indices = shuffled_indices[:train_set_size]
    val_indices = shuffled_indices[train_set_size:(train_set_size + val_set_size)]
    test_indices = shuffled_indices[(train_set_size + val_set_size):]
    return data[train_indices], data[val_indices], data[test_indices]


def stratified_sampling_mask(labels, num_class, train_ratio=0.6, val_ratio=0.2, random_seed=None):
    length = len(labels)
    train_mask, val_mask, test_mask = np.zeros(length), np.zeros(length), np.zeros(length)
    for i in range(num_class):
        indexs = np.where(labels == i)
        tra, val, tes = split_data(indexs[0], train_ratio, val_ratio, random_seed)
        train_mask[tra], val_mask[val], test_mask[tes] = 1, 1, 1
    return th.BoolTensor(train_mask), th.BoolTensor(val_mask), th.BoolTensor(test_mask)


def set_seed(seed=9699): # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    np.manual_seed(seed)
    #根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
    #但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    np.cuda.manual_seed(seed)
    #cuDNN在使用deterministic模式时（下面两行），可能会造成性能下降（取决于model）
    np.backends.cudnn.deterministic = True
    np.backends.cudnn.benchmark = False