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


device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
set_seed(42)

# data = citegrh.load_cora()
# num_feats, num_classes = data.features.shape[1], data.num_labels
# g, features, labels, train_mask, val_mask, test_mask = load_data_default(data)

# g, features, labels = load_data(data)
# train_mask, val_mask, test_mask = stratified_sampling_mask(data.labels, num_classes, 0.6, 0.2)

g, features, labels, train_mask, val_mask, test_mask, num_feats, num_classes = load_data_from_file('chameleon', None,
                                                                                                   0.6, 0.2)

# print_graph_info(g)
# g = cut_graph(g, labels, num_classes)
# g = g.to_networkx()
# g = DGLGraph(g)
# print_graph_info(g)

# erase_features(features, val_mask, test_mask, p=1)

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

test_losses = []
test_accs = []
num_hidden = 112


for i in range(4, 5):
    # model = ResGCNNet(num_feats, num_classes, num_hidden, i, bias=False, activation=F.tanh, graph_norm=True,
    #                   batch_norm=False, pair_norm=False, residual=False, dropout=0, dropedge=0, cutgraph=0,
    #                   init_beta=1., learn_beta=False)
    # model = DenseGCNNet(num_feats, num_classes, num_hidden, i, bias=False, activation=F.tanh, graph_norm=True,
    #                     batch_norm=False, dropout=0)
    # model = DglGCNNet(num_feats, num_classes, num_hidden, i, bias=False, activation=F.relu, graph_norm=True)
    # model = DglAGNNNet(num_feats, num_classes, i, bias=False)
    # model = ResAGNNNet(num_feats, num_classes, num_hidden, i, project=True, bias=False, activation=F.tanh,
    #                    init_beta=1., learn_beta=False,
    #                    batch_norm=False, residual=False, dropout=0, cutgraph=0.7)
    # model = ResGATNet(num_feats, num_classes, num_hidden, i, num_heads=1, merge='cat',
    #                   activation=F.elu, graph_norm=False, batch_norm=False, residual=False, dropout=0)
    # model = DenseGATNet(num_feats, num_classes, num_hidden, i, num_heads=1, merge='cat',
    # #                     activation=F.elu, graph_norm=False, batch_norm=True, dropout=0.5)
    # model = SGCLayer(num_feats, num_classes, i, cached=True, bias=False, graph_norm=True, pair_norm=False,
    #                  dropedge=0, cutgraph=0.8)
    # model = RootSGCNet(num_feats, num_classes, num_hidden, cached=True, bias=False, dropedge=0, cutgraph=0)
    # model = DglAPNNNet(num_feats, num_classes, i, alpha=1, bias=False, activation=None)
    # model = ResMLPNet(num_feats, num_classes, num_hidden, i, bias=False, activation=F.tanh, batch_norm=False,
    #                   residual=False, dropout=0)
    print(model)
    early_stopping = EarlyStopping(50, file_name="Try")
    optimizer = th.optim.Adam(model.parameters(), lr=1e-2)
    num_epoch = 400
    model = model.to(device)

    test_loss, test_acc = train_and_evaluate(num_epoch, model, optimizer, early_stopping, g, features, labels,
                                             train_mask, val_mask, test_mask)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

print(test_accs)

# plt.plot(test_accs)
# plt.show()
