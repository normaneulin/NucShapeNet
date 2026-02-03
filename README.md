# MLSNet / NucShapeNet (Initial Trial)

This repository includes the original MLSNet TFBS architectures and an initial trial under `NucShapeNet/` for nucleosome vs linker classification on D. melanogaster. The NucShapeNet trial reuses the MLSNet pipeline, adds DNA shape as auxiliary features from DNAshapeR, and uses 10‑fold cross‑validation.

## Dependencies

Tested with Python 3.8. Recommended packages:

- python==3.8.18
- pytorch==2.1.1
- numpy==1.24.1
- pandas==1.4.4
- scikit-learn==1.3.0
- tqdm (for progress bars)
- biopython (for FASTA parsing in preprocessing)

## Data & Preprocessing (NucShapeNet)

Input data for the trial lives in `NucShapeNet/Datasets`:

- Raw FASTA (sequence) at `NucShapeNet/Datasets/Raw Data/Sequence/nucleosomes_vs_linkers_melanogaster.fas`.
- Raw shape files (MGW, ProT, Roll, HelT, EP) at `NucShapeNet/Datasets/Raw Data/Shape/` produced by DNAshapeR.

Preprocessing scripts:

- `NucShapeNet/preprocess_seq_data.py` → writes `master_sequences.csv` to `NucShapeNet/Datasets/Master Data/Sequence/` with columns: `ID`, `Sequence`, `Label` (Label: 1 for nucleosome, 0 for linker).
- `NucShapeNet/preprocess_shape_data.py` → writes `master_{feature}.csv` for each of `MGW`, `ProT`, `Roll`, `HelT`, `EP` to `NucShapeNet/Datasets/Master Data/Shape/`.
  - Headers are `1..147`.
  - `NA` values are converted to `0.0`.
  - Step features (`Roll`, `HelT`) are zero‑padded to 147 columns.

Sequence encoding during training uses 3‑mer + one‑hot:

- A length‑147 sequence becomes overlapping 3‑mers with width `147 − 3 + 1 = 145`.
- One‑hot per base (A,C,G,T → 4 dims) gives a 12×145 matrix (12 = 4×3).

Shape features are stacked to a 5×147 matrix (features: MGW, ProT, Roll, HelT, EP).

## Model Adaptations (NucShapeNet)

- Sequence branch kernels are enlarged to 3×3, 7×7, and 11×11 (vs. smaller TFBS‑oriented kernels) to better capture nucleosome positional context.
- Attention/stoken module (`STVit`) is reused for the shape branch.
- Inputs expected by the model: sequence 12×145, shape 5×147.

## Training & Validation

- The NucShapeNet trial uses 10‑fold stratified cross‑validation (`sklearn.model_selection.StratifiedKFold`).
- Per‑fold training with an internal hold‑out validation split; metrics reported: ACC, ROC‑AUC, PR‑AUC.
- Models are saved under `NucShapeNet/save_models/` as `MLSNet_fold{k}.pth`.

## How to Run (NucShapeNet)

1. Generate master data:
	- `python NucShapeNet/preprocess_seq_data.py`
	- `python NucShapeNet/preprocess_shape_data.py`
2. Train with 10‑fold CV (flat module layout):
	- `cd NucShapeNet`
	- `python train.py`
	- Key modules now live at the NucShapeNet root: `MLSNet.py`, `STVit.py`, `DataReader.py`, `Embedding.py`, `Evaluator.py`, `Predict.py`.

## Notes

- Feature order for shapes in the trial is `['EP','HelT','MGW','ProT','Roll']`; keep it consistent across preprocessing and loading.
- If sequence lengths differ from 147, add pad/truncate logic before 3‑mer encoding to preserve 12×145 input size.

## Prediction Summary

After training, you can generate a per‑fold evaluation summary plot:

```
cd NucShapeNet
python Predict.py
```

Outputs go to `NucShapeNet/save_results/plots/`.
