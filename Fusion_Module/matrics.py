import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


@torch.no_grad()
def evaluate_with_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
):
    model.eval()
    all_y, all_pred, all_prob = [], [], []

    for x1, x2, y in loader:
        x1 = x1.to(device)
        x2 = x2.to(device)

        logits = model(x1, x2)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        pred = np.argmax(probs, axis=1)

        all_y.append(y.numpy())
        all_pred.append(pred)
        all_prob.append(probs)

    y_true = np.concatenate(all_y, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    y_prob = np.concatenate(all_prob, axis=0)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")

    auc = None
    try:
        if num_classes == 2:
            auc = roc_auc_score(y_true, y_prob[:, 1])
        else:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = None

    return acc, f1, auc, y_true, y_pred