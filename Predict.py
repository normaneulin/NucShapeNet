import os
import pickle
import csv
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from MLSNet import MLSNet
from DataReader import NucShapeNetDataset
import Evaluator as evaluator

SAVE_MODELS_DIR = os.path.join(os.path.abspath(os.curdir), 'save_models')
RESULTS_DIR = os.path.join(os.path.abspath(os.curdir), 'save_results')
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)


def evaluate_fold(model: nn.Module, test_indices: np.ndarray):
    ds = NucShapeNetDataset()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for i in test_indices:
            seq, shape, label = ds[i]
            out = model(seq.unsqueeze(0).to(device), shape.unsqueeze(0).to(device))
            preds.append(float(out.squeeze().cpu().item()))
            trues.append(int(label.squeeze().cpu().item()))
    y_score = np.array(preds, dtype=float)
    y_true = np.array(trues, dtype=int)
    y_pred = evaluator.pred2label(y_score)

    return {
        'ACC': evaluator.acc(y_true, y_pred),
        'Sn': evaluator.recall(y_true, y_pred),
        'Sp': evaluator.specificity(y_true, y_pred),
        'MCC': evaluator.mcc(y_true, y_pred),
        'F1_Score': evaluator.f1score(y_true, y_pred),
        'AUROC': evaluator.aucScore(y_true, y_score),
        'PR_AUC': evaluator.pr_auc(y_true, y_score),
        'Precision': evaluator.precision(y_true, y_pred),
    }


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load saved fold indices (train.py output)
    folds_path = os.path.join(RESULTS_DIR, 'folds.pickle')
    if not os.path.exists(folds_path):
        print(f"Error: folds.pickle not found at {folds_path}. Run training first.")
        return
    with open(folds_path, 'rb') as fp:
        folds = pickle.load(fp)

    metrics = {
        'ACC': [], 'Sn': [], 'Sp': [], 'MCC': [], 'F1_Score': [], 'AUROC': [], 'PR_AUC': [], 'Precision': []
    }
    processed_folds = []

    for fold_idx, fold in enumerate(folds, start=1):
        model = MLSNet().to(device)
        best_path = os.path.join(SAVE_MODELS_DIR, f"MLSNet_fold{fold_idx}_best.pth")
        path = os.path.join(SAVE_MODELS_DIR, f"MLSNet_fold{fold_idx}.pth")
        ckpt = best_path if os.path.exists(best_path) else path
        if not os.path.exists(ckpt):
            print(f"Warning: checkpoint not found for fold {fold_idx}: {ckpt}")
            continue
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state)

        m = evaluate_fold(model, fold['test_idx'])
        for k in metrics:
            metrics[k].append(m[k])
        processed_folds.append(fold_idx)
        print(
            f"Fold {fold_idx}: "
            f"Sn={m['Sn']:.4f} Sp={m['Sp']:.4f} ACC={m['ACC']:.4f} MCC={m['MCC']:.4f} "
            f"F1_Score={m['F1_Score']:.4f} AUROC={m['AUROC']:.4f} PR_AUC={m['PR_AUC']:.4f} Precision={m['Precision']:.4f}"
        )

    # Compute mean and std
    means = {k: float(np.nanmean(v)) for k, v in metrics.items()}
    stds = {k: float(np.nanstd(v)) for k, v in metrics.items()}

    # Plot summary bar chart (selected metrics)
    labels_order = ["ACC", "Sn", "Sp", "MCC", "AUROC", "PR_AUC", "F1_Score", "Precision"]
    x = np.arange(len(labels_order))
    vals = [means[label_name] for label_name in labels_order]
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111)
    ax.bar(x, vals, width=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_order)
    ax.set_ylim(0, 1.10)
    for i, label_name in enumerate(labels_order):
        ax.text(i - 0.40, vals[i] + 0.05, f"{vals[i]:.4f}±{stds[label_name]:.4f}", color='blue', fontweight='bold')
    out_png = os.path.join(PLOTS_DIR, "Predict_Evaluations.png")
    out_svg = os.path.join(PLOTS_DIR, "Predict_Evaluations.svg")
    out_eps = os.path.join(PLOTS_DIR, "Predict_Evaluations.eps")
    fig.savefig(out_png, bbox_inches='tight')
    fig.savefig(out_svg, format='svg', bbox_inches='tight')
    fig.savefig(out_eps, format='eps', bbox_inches='tight')
    print(f"Saved plots: {out_png}, {out_svg}, {out_eps}")

    # Save per-fold metrics CSV
    csv_path = os.path.join(RESULTS_DIR, 'predict_metrics.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Fold'] + labels_order)
        for i, fold_id in enumerate(processed_folds):
            row = [fold_id] + [metrics[label_name][i] for label_name in labels_order]
            writer.writerow(row)
        writer.writerow(['Mean'] + [means[label_name] for label_name in labels_order])
        writer.writerow(['Std'] + [stds[label_name] for label_name in labels_order])
    print(f"Saved metrics CSV: {csv_path}")


if __name__ == "__main__":
    main()
