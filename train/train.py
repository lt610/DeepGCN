import random

from utils.early_stopping import EarlyStopping
from utils.metic import evaluate_acc, evaluate
import time
import numpy as np
import torch.nn.functional as F
import torch as th
import torch.nn as nn


def train_net(num_epoch, model, optimizer, early_stopping, g, features, labels, train_mask, val_mask):
    dur = []
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    for epoch in range(num_epoch):
        if epoch >= 3:
            t0 = time.time()
        model.train()
        logits = model(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()

        # nn.utils.clip_grad_norm(model.parameters(), max_norm=5)

        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        train_acc = evaluate_acc(model, g, features, labels, train_mask)
        val_loss, val_acc = evaluate(model, g, features, labels, val_mask)
        train_losses.append(loss.item())
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        early_stopping(val_acc, model)
        print(
            "Epoch {:05d} | Loss {:.4f} | Train Acc {:.4f} | Val Loss {:.4f} | Val Acc {:.4f} | Time(s) {:.4f}".format(
                epoch, loss.item(), train_acc, val_loss, val_acc, np.mean(dur)))
        if early_stopping.is_stop:
            print("Early stopping")
            model.load_state_dict(early_stopping.load_checkpoint())
            break
    return train_losses, train_accs, val_losses, val_accs


def train_and_evaluate(num_epoch, model, optimizer, early_stopping, g, features, labels, train_mask, val_mask, test_mask):

    train_losses, train_accs, val_losses, val_accs = train_net(num_epoch, model, optimizer, early_stopping, g,
                                                               features, labels, train_mask, val_mask)
    test_loss, test_acc = evaluate(model, g, features, labels, test_mask)
    train_loss, train_acc = evaluate(model, g, features, labels, train_mask)
    print("Test Loss {:.4f} | Test Acc {:.4f}".format(test_loss, test_acc))
    print("max val acc:{:.4f}".format(max(val_accs)))
    print("epoch of max val acc:{}".format(val_accs.index(max(val_accs)) + 1))
    print("min val loss:{:.4f}".format(min(val_losses)))
    print("epoch of min val loss:{}".format(val_losses.index(min(val_losses)) + 1))
    print("best val score:{:.4f}".format(early_stopping.best_score))
    # print("index:{}".format(val_losses.index(-early_stopping.best_score)))
    # print("epoch of best val score:{}".format(val_accs.index(early_stopping.best_score)))

    # plt.plot(train_losses, color="b")
    # plt.plot(val_losses, color="r")
    # plt.show()
    # plt.plot(train_accs, color="b")
    # plt.plot(val_accs, color="r")
    # plt.show()
    return train_loss, train_acc, test_loss, test_acc


def grid_search():
    return


def set_seed(seed=9699):
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.backends.cudnn.deterministic = True
    th.backends.cudnn.benchmark = False

