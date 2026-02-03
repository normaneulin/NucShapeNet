import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
    confusion_matrix,
    matthews_corrcoef,
)


def build_kfold_indices(labels, k=10, shuffle=True, seed=0):
    """Return list of dicts with train/test indices for reproducible stratified k-fold splits."""
    labels = np.asarray(labels).squeeze()
    skf = StratifiedKFold(n_splits=k, shuffle=shuffle, random_state=seed)
    folds = []
    for train_index, test_index in skf.split(np.arange(len(labels)), labels):
        folds.append({
            'train_idx': np.array(train_index),
            'test_idx': np.array(test_index),
        })
    return folds


def pred2label(y_pred):
    return np.round(np.clip(y_pred, 0, 1))


def precision(y_true, y_pred):
    y_pred_bin = np.round(np.clip(y_pred, 0, 1))
    return precision_score(y_true, y_pred_bin)


def recall(y_true, y_pred):
    y_pred_bin = np.round(np.clip(y_pred, 0, 1))
    return recall_score(y_true, y_pred_bin)


def specificity(y_true, y_pred):
    y_pred_bin = np.round(np.clip(y_pred, 0, 1))
    cm = confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
    tn, fp = cm[0, 0], cm[0, 1]
    return (tn / (tn + fp)) if (tn + fp) > 0 else 0.0


def f1score(y_true, y_pred):
    y_pred_bin = np.round(np.clip(y_pred, 0, 1))
    return f1_score(y_true, y_pred_bin)


def aucScore(y_true, y_score):
    # ROC-AUC from probability scores
    try:
        return roc_auc_score(y_true, y_score)
    except ValueError:
        return float('nan')


def pr_auc(y_true, y_score):
    # Precision-Recall AUC from probability scores
    try:
        precision_arr, recall_arr, _ = precision_recall_curve(y_true, y_score)
        return auc(recall_arr, precision_arr)
    except ValueError:
        return float('nan')


def mcc(y_true, y_pred):
    y_pred_bin = np.round(np.clip(y_pred, 0, 1))
    try:
        return matthews_corrcoef(y_true, y_pred_bin)
    except ValueError:
        return 0.0


def acc(y_true, y_pred):
    y_pred_bin = np.round(np.clip(y_pred, 0, 1))
    return accuracy_score(y_true, y_pred_bin)
