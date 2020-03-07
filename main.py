from nets.res_gcn_net import ResGCNLayerNet
from dgl.data import citation_graph as citegrh
import torch as th

from train.train import train_net
from utils.data import load_data_default
from utils.early_stopping import EarlyStopping

model = ResGCNLayerNet(8)
print(model)
early_stopping = EarlyStopping(200)
# set_seed()
data = citegrh.load_cora()
g, features, labels, train_mask, val_mask, test_mask = load_data_default(data)
# g, features, labels = load_data(data)
# train_mask, val_mask, test_mask = stratified_sampling_mask(data.labels, data.num_labels, 0.6, 0.2, 42)
# optimizer = th.optim.Adam([
#                 {'params': net.gcn0.parameters()},
#                 {'params': net.gcns.parameters()},
#                 {'params': net.gcnk.parameters()}
#             ],lr=1e-2)
optimizer = th.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.01)
# optimizer = th.optim.Adam(net.parameters(), lr=1e-2, weight_decay=0.01)
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# g.to(device)
# features.to(device)
# labels.to(device)
# train_mask.to(device)
# test_mask.to(device)
model.to(device)
g = g.to(device)
features = features.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
test_mask = test_mask.to(device)
early_stopping = EarlyStopping(200)
num_epoch = 400
train_losses, train_accs, val_losses, val_accs = train_net(num_epoch , model, optimizer, early_stopping , g, features, labels, train_mask, val_mask)