import os
import math
import torch
import numpy as np
import torch.optim as optim
import torch.utils.data as loader
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, confusion_matrix, f1_score, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from models.MLSNet import MLSNet
from Datasets.DataReader import NucShapeNetDataset


class Trainer:
    def __init__(self, model_name='MLSNet'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.batch_size = 64
        self.epochs = 15
        self.loss_function = nn.BCELoss()
        self.learning_rate = 1e-3
        self.early_stop_patience = 5
        self.save_dir = os.path.join(os.path.abspath(os.curdir), 'save_models')
        os.makedirs(self.save_dir, exist_ok=True)
        self.results_dir = os.path.join(os.path.abspath(os.curdir), 'save_results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.metrics_file = os.path.join(self.results_dir, 'metrics.txt')

    def fit(self, model, train_loader, val_loader, best_save_path: str | None = None):
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5)
        train_epoch_losses = []
        best_val_loss = float('inf')
        epochs_no_improve = 0
        for epoch in range(self.epochs):
            model.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
            for seq, shape, label in pbar:
                optimizer.zero_grad()
                output = model(seq.to(self.device), shape.to(self.device))
                loss = self.loss_function(output, label.float().to(self.device))
                pbar.set_postfix(loss=loss.item())
                loss.backward()
                optimizer.step()
                train_epoch_losses.append(loss.item())
            # validation loss
            val_losses = []
            model.eval()
            with torch.no_grad():
                for vseq, vshape, vlabel in val_loader:
                    vout = model(vseq.to(self.device), vshape.to(self.device))
                    vlabel = vlabel.float().to(self.device)
                    val_losses.append(self.loss_function(vout, vlabel).item())
                val_loss_avg = torch.mean(torch.tensor(val_losses))
                scheduler.step(val_loss_avg)
            train_loss_avg = float(torch.mean(torch.tensor(train_epoch_losses))) if train_epoch_losses else float('nan')
            print(f"Epoch {epoch} — Train Loss: {train_loss_avg:.6f} | Val Loss: {float(val_loss_avg):.6f}")
            # save best checkpoint
            current_val = float(val_loss_avg)
            if current_val + 1e-8 < best_val_loss:
                best_val_loss = current_val
                epochs_no_improve = 0
                if best_save_path is not None:
                    torch.save(model.state_dict(), best_save_path)
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.early_stop_patience:
                    print(f"Early stopping triggered after {self.early_stop_patience} epochs without val loss improvement.")
                    break
            # reset accumulator per epoch
            train_epoch_losses = []

    def evaluate(self, model, test_loader, threshold: float = 0.5):
        preds = []  # probabilities (floats)
        trues = []  # true labels (ints)
        model.eval()
        with torch.no_grad():
            for seq, shape, label in test_loader:
                out = model(seq.to(self.device), shape.to(self.device))
                # ensure scalar float and int
                preds.append(float(out.squeeze().detach().cpu().item()))
                trues.append(int(label.squeeze().detach().cpu().item()))
        # Convert to numpy arrays
        y_score = np.array(preds, dtype=float)
        y_true = np.array(trues, dtype=int)
        y_pred = (y_score >= threshold).astype(int)

        # Confusion matrix (force 2x2)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
        # Metrics
        acc = accuracy_score(y_true=y_true, y_pred=y_pred)
        sn = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity/recall
        sp = tn / (tn + fp) if (tn + fp) > 0 else 0.0  # specificity
        mcc = matthews_corrcoef(y_true=y_true, y_pred=y_pred) if ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) > 0 else 0.0
        f1 = f1_score(y_true=y_true, y_pred=y_pred) if len(np.unique(y_true)) > 1 else 0.0
        auroc = roc_auc_score(y_true=y_true, y_score=y_score) if len(np.unique(y_true)) > 1 else float('nan')

        if len(np.unique(y_true)) > 1:
            # Use positional args to avoid signature mismatches across sklearn versions
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            pr_auc = auc(recall, precision)
        else:
            pr_auc = float('nan')

        return {
            'ACC': acc,
            'Sn': sn,
            'Sp': sp,
            'MCC': mcc,
            'F1_Score': f1,
            'AUROC': auroc,
            'PR_AUC': pr_auc
        }

    def cross_validate(self, n_splits=10, random_state=0):
        # Load full dataset once
        full_ds = NucShapeNetDataset()
        labels = torch.tensor([full_ds.labels[i][0] for i in range(len(full_ds))])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_metrics = []
        # Initialize metrics file
        with open(self.metrics_file, 'w') as f:
            f.write(f"k={n_splits}\n")
        for fold, (train_idx, test_idx) in enumerate(skf.split(torch.arange(len(full_ds)), labels)):
            print(f"Fold {fold+1}/{n_splits}")
            # Further split train into train/val (hold-out within fold)
            # 10% of train for validation
            val_size = max(1, math.ceil(len(train_idx) * 0.1))
            train_indices = train_idx[:-val_size]
            val_indices = train_idx[-val_size:]

            train_ds = NucShapeNetDataset(indices=train_indices)
            val_ds = NucShapeNetDataset(indices=val_indices)
            test_ds = NucShapeNetDataset(indices=test_idx)

            train_loader = loader.DataLoader(train_ds, batch_size=self.batch_size, shuffle=True, drop_last=True)
            val_loader = loader.DataLoader(val_ds, batch_size=self.batch_size, shuffle=False, drop_last=False)
            test_loader = loader.DataLoader(test_ds, batch_size=1, shuffle=False, drop_last=False)

            model = MLSNet().to(self.device)
            # Define best checkpoint path for this fold
            best_save_path = os.path.join(self.save_dir, f"{self.model_name}_fold{fold+1}_best.pth")
            self.fit(model, train_loader, val_loader, best_save_path=best_save_path)

            # Save model per fold
            save_path = os.path.join(self.save_dir, f"{self.model_name}_fold{fold+1}.pth")
            torch.save(model.state_dict(), save_path)
            # Load best checkpoint for evaluation
            if os.path.exists(best_save_path):
                model.load_state_dict(torch.load(best_save_path, map_location=self.device))

            metrics = self.evaluate(model, test_loader)
            print(
                f"Fold {fold+1} "
                f"Sn={metrics['Sn']:.4f} Sp={metrics['Sp']:.4f} "
                f"ACC={metrics['ACC']:.4f} MCC={metrics['MCC']:.4f} "
                f"F1_Score={metrics['F1_Score']:.4f} AUROC={metrics['AUROC']:.4f} PR-AUC={metrics['PR_AUC']:.4f}"
            )
            with open(self.metrics_file, 'a') as f:
                f.write(
                    f"Fold {fold+1}: "
                    f"Sn={metrics['Sn']:.6f}, Sp={metrics['Sp']:.6f}, "
                    f"ACC={metrics['ACC']:.6f}, MCC={metrics['MCC']:.6f}, "
                    f"F1_Score={metrics['F1_Score']:.6f}, AUROC={metrics['AUROC']:.6f}, PR_AUC={metrics['PR_AUC']:.6f}\n"
                )
            fold_metrics.append(metrics)

        # Aggregate
        accs = [m['ACC'] for m in fold_metrics]
        rocs = [m['AUROC'] for m in fold_metrics]
        prs = [m['PR_AUC'] for m in fold_metrics]
        sns = [m['Sn'] for m in fold_metrics]
        sps = [m['Sp'] for m in fold_metrics]
        mccs = [m['MCC'] for m in fold_metrics]
        f1s = [m['F1_Score'] for m in fold_metrics]
        avg_line = (
            f"10-fold Averages: "
            f"Sn={sum(sns)/len(sns):.4f} Sp={sum(sps)/len(sps):.4f} "
            f"ACC={sum(accs)/len(accs):.4f} MCC={sum(mccs)/len(mccs):.4f} "
            f"F1_Score={sum(f1s)/len(f1s):.4f} AUROC={sum(rocs)/len(rocs):.4f} PR-AUC={sum(prs)/len(prs):.4f}"
        )
        print(avg_line)
        with open(self.metrics_file, 'a') as f:
            f.write(avg_line + "\n")
        return fold_metrics


if __name__ == "__main__":
    trainer = Trainer(model_name='MLSNet')
    trainer.cross_validate(n_splits=10, random_state=0)
