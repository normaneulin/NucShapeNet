import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from Initialization.Embedding import one_hot

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_SEQ_PATH = os.path.join(SCRIPT_DIR, "..", "Datasets", "Master Data", "Sequence", "master_sequences.csv")
MASTER_SHAPE_DIR = os.path.join(SCRIPT_DIR, "..", "Datasets", "Master Data", "Shape")

FEATURE_ORDER = ['EP', 'HelT', 'MGW', 'ProT', 'Roll']


def load_sequences(k=3):
    df = pd.read_csv(MASTER_SEQ_PATH)
    ids = df['ID'].tolist()
    seqs = df['Sequence'].tolist()
    labels = df['Label'].astype(int).to_numpy().reshape(-1, 1)

    num_samples = len(seqs)
    # For L=147 and k=3, width = L - k + 1 = 145; height = 4*k = 12
    encoded = np.empty((num_samples, 12, len(seqs[0]) - k + 1), dtype=float)
    for i, s in enumerate(seqs):
        mat = one_hot(s, k)  # (L-k+1, 4k)
        encoded[i] = np.transpose(mat, (1, 0))  # (4k, L-k+1) => (12, 145 when L=147)
    return ids, encoded, labels


def load_shapes(feature_order=FEATURE_ORDER):
    # Each master_{feature}.csv has columns 1..147
    shape_mats = []
    num_samples = None
    for feat in feature_order:
        path = os.path.join(MASTER_SHAPE_DIR, f"master_{feat}.csv")
        df = pd.read_csv(path)
        if num_samples is None:
            num_samples = df.shape[0]
        else:
            num_samples = min(num_samples, df.shape[0])
        shape_mats.append(df.iloc[:num_samples].to_numpy(dtype=float))  # (N,147)
    # Stack to (N, F, 147)
    stacked = np.stack(shape_mats, axis=1)
    return stacked


class NucShapeNetDataset(Dataset):
    def __init__(self, indices=None, k=3, feature_order=FEATURE_ORDER):
        self.ids, self.seqs, self.labels = load_sequences(k)
        self.shapes = load_shapes(feature_order)
        # Align lengths if needed
        n = min(self.seqs.shape[0], self.shapes.shape[0], self.labels.shape[0])
        self.ids = self.ids[:n]
        self.seqs = self.seqs[:n]
        self.shapes = self.shapes[:n]
        self.labels = self.labels[:n]
        self.indices = indices if indices is not None else np.arange(n)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        seq = torch.from_numpy(self.seqs[i])  # (12,145) when L=147, k=3
        shape = torch.from_numpy(self.shapes[i])  # (5,147)
        label = torch.tensor(self.labels[i], dtype=torch.float32)  # (1,)
        return seq, shape, label
