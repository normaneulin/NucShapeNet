import os
import math
import torch
import torch.optim as optim
import torch.utils.data as loader
from torch import nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
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
        self.save_dir = os.path.join(os.path.abspath(os.curdir), 'save_models')
        os.makedirs(self.save_dir, exist_ok=True)

    def fit(self, model, train_loader, val_loader):
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, verbose=1)
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

    def evaluate(self, model, test_loader):
        preds = []
        trues = []
        model.eval()
        with torch.no_grad():
            for seq, shape, label in test_loader:
                out = model(seq.to(self.device), shape.to(self.device))
                preds.append(out.squeeze().detach().cpu().numpy())
                trues.append(label.squeeze().detach().cpu().numpy())
        accuracy = accuracy_score(y_pred=(torch.tensor(preds).numpy().round()), y_true=trues)
        roc_auc = roc_auc_score(y_score=preds, y_true=trues)
        precision, recall, _ = precision_recall_curve(probas_pred=preds, y_true=trues)
        pr_auc = auc(recall, precision)
        return accuracy, roc_auc, pr_auc

    def cross_validate(self, n_splits=10, random_state=0):
        # Load full dataset once
        full_ds = NucShapeNetDataset()
        labels = torch.tensor([full_ds.labels[i][0] for i in range(len(full_ds))])
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        fold_metrics = []
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
            self.fit(model, train_loader, val_loader)

            # Save model per fold
            save_path = os.path.join(self.save_dir, f"{self.model_name}_fold{fold+1}.pth")
            torch.save(model.state_dict(), save_path)

            acc, roc, pr = self.evaluate(model, test_loader)
            print(f"Fold {fold+1} ACC={acc:.4f} ROC-AUC={roc:.4f} PR-AUC={pr:.4f}")
            fold_metrics.append((acc, roc, pr))

        # Aggregate
        accs = [m[0] for m in fold_metrics]
        rocs = [m[1] for m in fold_metrics]
        prs = [m[2] for m in fold_metrics]
        print(f"10-fold Average ACC={sum(accs)/len(accs):.4f} ROC-AUC={sum(rocs)/len(rocs):.4f} PR-AUC={sum(prs)/len(prs):.4f}")
        return fold_metrics


if __name__ == "__main__":
    trainer = Trainer(model_name='MLSNet')
    trainer.cross_validate(n_splits=10, random_state=0)
