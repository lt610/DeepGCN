from nets.res_gcn_net import ResGCNLayerNet
from dgl.data import citation_graph as citegrh
import torch as th
from train.train import train_and_evaluate, set_seed
from utils.data import load_data_default, load_data, stratified_sampling_mask
from utils.early_stopping import EarlyStopping
import matplotlib.pyplot as plt


device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
# set_seed(42)
data = citegrh.load_cora()
# g, features, labels, train_mask, val_mask, test_mask = load_data_default(data)
g, features, labels = load_data(data)
train_mask, val_mask, test_mask = stratified_sampling_mask(data.labels, data.num_labels, 0.6, 0.2)
# optimizer = th.optim.Adam([
#                 {'params': net.gcn0.parameters()},
#                 {'params': net.gcns.parameters()},
#                 {'params': net.gcnk.parameters()}
#             ],lr=1e-2)
g = g.to(device)
features = features.to(device)
labels = labels.to(device)
train_mask = train_mask.to(device)
val_mask = val_mask.to(device)
test_mask = test_mask.to(device)

test_losses = []
test_accs = []

for i in range(4, 9):
    model = ResGCNLayerNet(i)
    print(model)
    early_stopping = EarlyStopping(100, file_name="ResGCNLayerNet")
    optimizer = th.optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.01)
    num_epoch = 400
    model = model.to(device)
    test_loss, test_acc = train_and_evaluate(num_epoch, model, optimizer, early_stopping, g, features, labels, train_mask, val_mask, test_mask)
    test_losses.append(test_loss)
    test_accs.append(test_acc)


print(test_accs)
plt.plot(test_accs)
plt.show()
