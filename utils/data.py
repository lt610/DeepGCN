import numpy as np
import networkx as nx
import torch as th
from dgl import DGLGraph
import matplotlib.pyplot as plt


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


def stratified_sampling_mask(labels, num_classes, train_ratio=0.6, val_ratio=0.2, random_seed=None):
    length = len(labels)
    train_mask, val_mask, test_mask = np.zeros(length), np.zeros(length), np.zeros(length)
    for i in range(num_classes):
        indexes = np.where(labels == i)
        tra, val, tes = split_data(indexes[0], train_ratio, val_ratio, random_seed)
        train_mask[tra], val_mask[val], test_mask[tes] = 1, 1, 1
    return th.BoolTensor(train_mask), th.BoolTensor(val_mask), th.BoolTensor(test_mask)


def erase_features(features, val_mask, test_mask, p=0, random_seed=None):
    if p != 0:
        if p == 1:
            features[val_mask] = 0
            features[test_mask] = 0
        else:
            if random_seed:
                np.random.seed(random_seed)
            val_indexes = np.where(val_mask == True)[0]
            test_indexs = np.where(test_mask == True)[0]
            shuffled_val_indices = np.random.permutation(len(val_indexes))
            shuffled_test_indices = np.random.permutation(len(test_indexs))
            val_erase = shuffled_val_indices[int(len(val_indexes) * p)]
            test_erase = shuffled_test_indices[int(len(test_indexs) * p)]
            features[val_erase] = 0
            features[test_erase] = 0


def print_data_info(data):
    print('  NumNodes: {}'.format(data.graph.number_of_nodes()))
    print('  NumEdges: {}'.format(data.graph.number_of_edges()))
    print('  NumFeats: {}'.format(data.features.shape[1]))
    print('  NumClasses: {}'.format(data.num_labels))
    print('  NumTrainingSamples: {}'.format(len(np.nonzero(data.train_mask)[0])))
    print('  NumValidationSamples: {}'.format(len(np.nonzero(data.val_mask)[0])))
    print('  NumTestSamples: {}'.format(len(np.nonzero(data.test_mask)[0])))


def print_graph_info(graph):
    print("number of nodes:{}".format(graph.number_of_nodes()))
    print("number of edges:{}".format(graph.number_of_edges()))


def draw_part_graph(graph, nodes=None):
    if nodes is not None:
        g = graph.subgraph(nodes)
    else:
        g = graph
    pos = nx.kamada_kawai_layout(g)
    nx.draw(g, pos, with_labels=True, node_color=[[.7, .7, .7]])
    plt.show()


def cut_graph(graph, labels, num_classes):
    graph = graph.local_var()
    length1 = len(labels)
    indexes = []
    for i in range(num_classes):
        index = np.where(labels == i)
        t = np.zeros(length1)
        t[index[0]] = 1
        indexes.append(t)
    edges = graph.edges()
    length2 = len(edges[0])
    delete_edges = []
    for i in range(length2):
        u, v = edges[0][i], edges[1][i]
        cla = labels[u]
        if indexes[cla][v] == 0:
            delete_edges.append(i)
    graph.remove_edges(delete_edges)
    return graph


def draw(train_accs0, train_accs1, test_accs0, test_accs1):
    return
