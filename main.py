import itertools

from layers.sgc_layer import SGCLayer
from nets.dense_gat_net import DenseGATNet
from nets.dense_gcn_net import DenseGCNNet
from nets.dgl_appnp_net import DglAPNNNet
from nets.dgl_gcn_net import DglGCNNet
from nets.res_agnn_net import ResAGNNNet
from nets.res_gat_net import ResGATNet
from nets.res_gcn_net import ResGCNNet
from dgl.data import citation_graph as citegrh
import torch as th

from nets.res_mlp_net import ResMLPNet
from nets.root_sgc_net import RootSGCNet
from train.train import train_and_evaluate, set_seed
from utils.data import load_data_default, load_data, stratified_sampling_mask, cut_graph, print_graph_info, \
    erase_features
from utils.early_stopping import EarlyStopping
import matplotlib.pyplot as plt
import torch.nn.functional as F
from dgl import DGLGraph
from utils.data_other import load_data_from_file
from dgl.nn.pytorch import SGConv, APPNPConv
from nets.dgl_agnn_net import DglAGNNNet
from sklearn.manifold import Isomap
import numpy as np


def train1(data_params, model_params):
    device, g, features, labels, train_mask, val_mask, test_mask, num_feats, \
        num_classes = data_params['device'], data_params['g'], data_params['features'], \
        data_params['labels'], data_params['train_mask'], data_params['val_mask'], \
        data_params['test_mask'], data_params['num_feats'], data_params['num_classes']
    num_hidden, layers, bias, activation, graph_norm, batch_norm, pair_norm, residual,\
        dropout, dropedge, cutgraph = model_params['num_hidden'], model_params['layers'], model_params['bias'], \
        model_params['activation'], model_params['graph_norm'], model_params['batch_norm'], model_params['pair_norm'],\
        model_params['residual'], model_params['dropout'], model_params['dropedge'], model_params['cutgraph']
    model_name = model_params['model_name']
    learn_rate, weight_decay = model_params['learn_rate'], model_params['weight_decay']
    if model_name == 'ResMLPNet':
        model = ResMLPNet(num_feats, num_classes, num_hidden, layers, bias=bias, activation=activation,
                          batch_norm=batch_norm, residual=residual, dropout=dropout)
    if model_name == 'SGCLayer':
        model = SGCLayer(num_feats, num_classes, layers, cached=True, bias=bias, graph_norm=graph_norm,
                         pair_norm=pair_norm, dropedge=dropedge, cutgraph=cutgraph)
    if model_name == 'RootSGCNet':
        model = RootSGCNet(num_feats, num_classes, num_hidden, cached=True, bias=bias, dropedge=dropedge,
                           cutgraph=cutgraph)
    if model_name == 'ResGCNNet':
        model = ResGCNNet(num_feats, num_classes, num_hidden, layers, bias=bias, activation=activation,
                          graph_norm=graph_norm, batch_norm=batch_norm, pair_norm=pair_norm,
                          residual=residual, dropout=dropout, dropedge=dropedge, cutgraph=cutgraph,
                          init_beta=1., learn_beta=False)
    if model_name == 'DenseGCNNet':
        model = DenseGCNNet(num_feats, num_classes, num_hidden, layers, bias=bias, activation=activation,
                            graph_norm=graph_norm, batch_norm=batch_norm, dropout=dropout)
    # model = DglGCNNet(num_feats, num_classes, num_hidden, layers, bias=False, activation=F.relu, graph_norm=True)
    # model = DglAGNNNet(num_feats, num_classes, layers, bias=False)
    if model_name == 'ResGATNet':
        model = ResGATNet(num_feats, num_classes, num_hidden, layers, num_heads=1, merge='cat',
                          activation=F.elu, graph_norm=graph_norm, batch_norm=batch_norm,
                          residual=residual, dropout=dropout)
    if model_name == 'DenseGATNet':
        model = DenseGATNet(num_feats, num_classes, num_hidden, layers, num_heads=1, merge='cat',
                            activation=F.elu, graph_norm=graph_norm, batch_norm=batch_norm, dropout=dropout)
    if model_name == 'DglAPNNNet':
        model = DglAPNNNet(num_feats, num_classes, layers, alpha=1, bias=bias, activation=None)
    if model_name == 'ResAGNNNet':
        model = ResAGNNNet(num_feats, num_classes, num_hidden, layers, project=True, bias=bias, activation=activation,
                           init_beta=1., learn_beta=False,
                           batch_norm=batch_norm, residual=residual, dropout=dropout, cutgraph=cutgraph)
    print(model)
    early_stopping = EarlyStopping(50, file_name="Try")
    optimizer = th.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=weight_decay)
    num_epoch = 400
    model = model.to(device)
    test_loss, test_acc = train_and_evaluate(num_epoch, model, optimizer, early_stopping, g, features, labels,
                                             train_mask, val_mask, test_mask)
    return test_loss, test_acc


def train2():
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    # set_seed(42)
    data_params = {}
    model_params = {}
    dataset_name = ['chameleon']
    model_name = ['SGCLayer']
    num_hidden = [128]
    layers = [i for i in range(2, 10)]

    model_params['bias'] = False
    model_params['activation'] = F.tanh
    model_params['graph_norm'] = True
    model_params['batch_norm'] = False
    model_params['pair_norm'] = False
    model_params['residual'] = False

    dropout = [0]
    dropedge = [0]
    cutgraph = [0]
    learn_rate = [1e-2]
    weight_decay = [0]

    params = itertools.product(dataset_name, model_name, num_hidden, layers, dropout, dropedge, cutgraph,
                               learn_rate, weight_decay)
    pre_dataset = ''
    losses = []
    acces = []
    for param in params:
        model_params['dataset_name'], model_params['model_name'], model_params['num_hidden'],\
            model_params['layers'], model_params['dropout'], model_params['dropedge'],\
            model_params['cutgraph'], model_params['learn_rate'], model_params['weight_decay'] = param[0],\
            param[1], param[2], param[3], param[4], param[5], param[6], param[7], param[8]
        if model_params['dataset_name'] != pre_dataset:
            pre_dataset = model_params['dataset_name']
            # data = citegrh.load_cora()
            # num_feats, num_classes = data.features.shape[1], data.num_labels
            # g, features, labels, train_mask, val_mask, test_mask = load_data_default(data)

            # g, features, labels = load_data(data)
            # train_mask, val_mask, test_mask = stratified_sampling_mask(data.labels, num_classes, 0.6, 0.2)

            g, features, labels, train_mask, val_mask, test_mask, num_feats,\
                num_classes = load_data_from_file(model_params['dataset_name'], None, 0.6, 0.2)
            train_mask, val_mask, test_mask = stratified_sampling_mask(labels, num_classes, 0.6, 0.2, random_seed=42)

            # print_graph_info(g)
            # g = cut_graph(g, labels, num_classes)
            # g = g.to_networkx()
            # g = DGLGraph(g)
            # print_graph_info(g)

            erase_features(features, val_mask, test_mask, p=0)

            # optimizer = th.optim.Adam([
            #                 {'params': net.gcn0.parameters()},
            #                 {'params': net.gcns.parameters()},
            #                 {'params': net.gcnk.parameters()}
            #             ],lr=1e-2)

            # n_components = int(num_feats * 0.8)
            # isomap = Isomap(n_components=n_components, n_neighbors=100)
            # features = th.FloatTensor(isomap.fit_transform(features))
            # num_feats = n_components

            g = g.to(device)
            features = features.to(device)
            labels = labels.to(device)
            train_mask = train_mask.to(device)
            val_mask = val_mask.to(device)
            test_mask = test_mask.to(device)

            data_params['device'], data_params['g'], data_params['features'], data_params['labels'],\
                data_params['train_mask'], data_params['val_mask'], data_params['test_mask'],\
                data_params['num_feats'], data_params['num_classes'] = device, g, features, labels,\
                train_mask, val_mask, test_mask, num_feats, num_classes
        losses1 = []
        acces1 = []
        for i in range(3):
            test_loss, test_acc = train1(data_params, model_params)
            losses1.append(test_loss)
            acces1.append(test_acc)
        # print(losses1)
        # print(acces1)
        loss = np.mean(losses1)
        acc = np.mean(acces1)
        losses.append(loss)
        acces.append(acc)
    print('loss:{}'.format(losses))
    print('acc:{}'.format(acces))


if __name__ == '__main__':
    train2()