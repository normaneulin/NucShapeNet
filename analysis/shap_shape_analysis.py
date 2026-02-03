import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import shap

# Ensure imports work when running from project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
sys.path.append(PROJECT_ROOT)

from MLSNet import MLSNet

# Keep consistent with README notes and data loading
FEATURE_ORDER = ["EP", "HelT", "MGW", "ProT", "Roll"]

class ShapeOnlyModel(nn.Module):
    """
    Wrapper around MLSNet to analyze SHAP values for shape inputs only.
    Provides a fixed baseline for the sequence branch (zeros), so SHAP attributions
    measure contributions of shape channels/positions relative to a neutral sequence.
    """
    def __init__(self, core_model: nn.Module, seq_height: int = 12, seq_width: int = 145, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.core = core_model
        self.device = device
        self.register_buffer('seq_baseline', torch.zeros(1, seq_height, seq_width, dtype=torch.float32, device=device))

    def forward(self, shape: torch.Tensor) -> torch.Tensor:
        # shape: (B, 5, 147)
        b = shape.shape[0]
        # Expand baseline to match batch
        seq = self.seq_baseline.expand(b, -1, -1)
        return self.core(seq, shape)


def load_shapes(data_dir: str, feature_order=FEATURE_ORDER) -> np.ndarray:
    """Load stacked shape features from Master Data/Shape into array (N, 5, 147)."""
    shape_dir = os.path.join(data_dir, 'Datasets', 'Master Data', 'Shape')
    feat_arrays = []
    for feat in feature_order:
        path = os.path.join(shape_dir, f"master_{feat}.csv")
        df = pd.read_csv(path)
        # Expect headers 1..147; convert to numpy
        feat_arrays.append(df.values.astype(np.float32))
    # Stack along channel: (N, 5, 147)
    stacked = np.stack(feat_arrays, axis=1)
    return stacked


def select_samples(x: np.ndarray, num_samples: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.shape[0], size=min(num_samples, x.shape[0]), replace=False)
    return x[idx]


def robust_load_checkpoint(model: nn.Module, checkpoint_path: str, device: torch.device) -> None:
    obj = torch.load(checkpoint_path, map_location=device)
    if isinstance(obj, dict) and 'model_state_dict' in obj:
        model.load_state_dict(obj['model_state_dict'])
    elif isinstance(obj, dict):
        # Try loading as state_dict directly
        try:
            model.load_state_dict(obj)
        except Exception as e:
            raise RuntimeError(f"Unsupported checkpoint format: keys={list(obj.keys())}") from e
    else:
        raise RuntimeError("Checkpoint is not a supported dict format.")


def main():
    parser = argparse.ArgumentParser(description="SHAP analysis for 5 shape features on MLSNet.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to trained model checkpoint (.pth)')
    parser.add_argument('--data-root', type=str, default=PROJECT_ROOT, help='Project root containing Datasets/Master Data')
    parser.add_argument('--output-dir', type=str, default=os.path.join(PROJECT_ROOT, 'Finetuning Documentations', 'trial1_10_folds', 'shap_analysis'), help='Directory to save SHAP outputs')
    parser.add_argument('--background-size', type=int, default=100, help='Number of background examples for SHAP')
    parser.add_argument('--sample-size', type=int, default=200, help='Number of evaluation examples to explain')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load shapes
    shapes = load_shapes(args.data_root)
    bg_np = select_samples(shapes, args.background_size, seed=args.seed)
    eval_np = select_samples(shapes, args.sample_size, seed=args.seed + 1)

    # Prepare tensors (B, 5, 147)
    bg = torch.from_numpy(bg_np).to(device)
    eval_t = torch.from_numpy(eval_np).to(device)

    # Build and load model
    core = MLSNet().to(device)
    core.eval()
    robust_load_checkpoint(core, args.checkpoint, device)

    shape_model = ShapeOnlyModel(core_model=core, device=device)

    # SHAP GradientExplainer (fast, model-specific)
    explainer = shap.GradientExplainer(shape_model, bg)
    shap_vals = explainer.shap_values(eval_t)
    # For binary output, shap returns a list with one array
    if isinstance(shap_vals, list):
        shap_arr = shap_vals[0]
    else:
        shap_arr = shap_vals
    # shap_arr: (num_samples, 5, 147)

    # Aggregate global importance by shape type
    abs_vals = np.abs(shap_arr)
    per_type_importance = abs_vals.mean(axis=(0, 2))  # (5,)
    summary_df = pd.DataFrame({
        'Feature': FEATURE_ORDER,
        'MeanAbsSHAP': per_type_importance
    })
    summary_path = os.path.join(args.output_dir, 'shape_type_importance.csv')
    summary_df.to_csv(summary_path, index=False)

    # Per-position heatmaps (mean over samples)
    per_pos = abs_vals.mean(axis=0)  # (5, 147)
    pos_cols = [str(i) for i in range(1, 148)]
    heatmap_df = pd.DataFrame(per_pos, index=FEATURE_ORDER, columns=pos_cols)
    heatmap_path = os.path.join(args.output_dir, 'shape_position_heatmap.csv')
    heatmap_df.to_csv(heatmap_path)

    # Save raw SHAP values subset for optional deeper dives
    raw_path = os.path.join(args.output_dir, 'shap_values_subset.npy')
    np.save(raw_path, shap_arr)

    print(f"Saved: {summary_path}\nSaved: {heatmap_path}\nSaved raw SHAP: {raw_path}")


if __name__ == '__main__':
    main()
