import os
import math
import argparse
import pickle
import torch
import numpy as np
import torch.optim as optim
import torch.utils.data as loader
from torch import nn
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

from MLSNet import MLSNet
from DataReader import NucShapeNetDataset
import Evaluator as evaluator


class Trainer:
    def __init__(self, model_name='MLSNet', n_splits: int = 10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.n_splits = n_splits
        self.batch_size = 64
        self.epochs = 15
        self.loss_function = nn.BCELoss()
        self.learning_rate = 1e-3
        self.early_stop_patience = 5
        self.save_dir = os.path.join(os.path.abspath(os.curdir), 'save_models')
        os.makedirs(self.save_dir, exist_ok=True)
        self.results_dir = os.path.join(os.path.abspath(os.curdir), 'save_results')
        os.makedirs(self.results_dir, exist_ok=True)
        self.loss_file = os.path.join(self.results_dir, 'loss.txt')

    def fit(self, model, train_loader, val_loader, best_save_path: str | None = None, fold_id: int = 0):
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

            # log losses per epoch per fold
            with open(self.loss_file, 'a') as lf:
                lf.write(f"Fold {fold_id}, Epoch {epoch}, Train_Loss={train_loss_avg:.6f}, Val_Loss={float(val_loss_avg):.6f}\n")
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

    def cross_validate(self, n_splits: int | None = None, random_state: int = 0):
        # Resolve number of splits from Trainer if not provided
        if n_splits is None:
            n_splits = self.n_splits
        # Load full dataset once
        full_ds = NucShapeNetDataset()
        labels = torch.tensor([full_ds.labels[i][0] for i in range(len(full_ds))])
        # Build folds via Evaluator for parity with the requested layout
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        # Initialize loss file header
        with open(self.loss_file, 'w') as lf:
            lf.write(f"k={n_splits}\n")

        # Save fold indices for Predict.py to reuse
        folds = []
        for fold, (train_idx, test_idx) in enumerate(skf.split(torch.arange(len(full_ds)), labels)):
            print(f"Fold {fold+1}/{n_splits}")
            folds.append({
                'train_idx': np.array(train_idx),
                'test_idx': np.array(test_idx)
            })
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
            self.fit(model, train_loader, val_loader, best_save_path=best_save_path, fold_id=fold+1)

            # Save model per fold
            save_path = os.path.join(self.save_dir, f"{self.model_name}_fold{fold+1}.pth")
            torch.save(model.state_dict(), save_path)
        # Persist fold indices for Predict.py
        folds_path = os.path.join(self.results_dir, 'folds.pickle')
        with open(folds_path, 'wb') as fp:
            pickle.dump(folds, fp)
        print(f"Saved fold indices to {folds_path}")
        return folds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MLSNet with stratified k-fold CV")
    parser.add_argument("--kfolds", type=int, default=10, help="Number of stratified folds")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for StratifiedKFold")
    args = parser.parse_args()

    trainer = Trainer(model_name='MLSNet', n_splits=args.kfolds)
    trainer.cross_validate(random_state=args.seed)  
