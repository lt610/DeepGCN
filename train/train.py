from utils.early_stopping import EarlyStopping
from utils.metic import evaluate_acc, evaluate
import time
import numpy as np
import torch.nn.functional as F


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

