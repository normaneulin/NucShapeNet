import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
# 1. Set a seed so the shuffle is reproducible (you get the same shuffle every time)
RANDOM_SEED = 42

# 2. Define Base Paths (Using raw strings r"..." for Windows paths)
BASE_DIR = r"C:\Software Projects\MLSNet\NucShapeNet\Datasets\Master Data"

# Input Files (Unshuffled)
INPUT_FILES = {
    "Sequence": os.path.join(BASE_DIR, "Sequence", "master_sequences_unshuffled.csv"),
    "EP":       os.path.join(BASE_DIR, "Shape", "master_EP_unshuffled.csv"),
    "HelT":     os.path.join(BASE_DIR, "Shape", "master_HelT_unshuffled.csv"),
    "MGW":      os.path.join(BASE_DIR, "Shape", "master_MGW_unshuffled.csv"),
    "ProT":     os.path.join(BASE_DIR, "Shape", "master_ProT_unshuffled.csv"),
    "Roll":     os.path.join(BASE_DIR, "Shape", "master_Roll_unshuffled.csv"),
}

# Output Files (Shuffled)
OUTPUT_FILES = {
    "Sequence": os.path.join(BASE_DIR, "Sequence", "master_sequences.csv"),
    "EP":       os.path.join(BASE_DIR, "Shape", "master_EP.csv"),
    "HelT":     os.path.join(BASE_DIR, "Shape", "master_HelT.csv"),
    "MGW":      os.path.join(BASE_DIR, "Shape", "master_MGW.csv"),
    "ProT":     os.path.join(BASE_DIR, "Shape", "master_ProT.csv"),
    "Roll":     os.path.join(BASE_DIR, "Shape", "master_Roll.csv"),
}

def main():
    print("--- Starting Synchronized Shuffle ---")
    
    # 1. Load All DataFrames
    dfs = {}
    print("Loading files...")
    first_len = None
    # Expected headers for Shape files and Sequence file
    expected_shape_headers = [str(i) for i in range(1, 148)]
    expected_seq_columns = {"ID", "Sequence", "Label"}
    
    for key, path in INPUT_FILES.items():
        if not os.path.exists(path):
            print(f"CRITICAL ERROR: File not found: {path}")
            return
            
        df = pd.read_csv(path)
        dfs[key] = df
        
        # Check that all files have the same number of rows
        if first_len is None:
            first_len = len(df)
            print(f"  -> {key} loaded ({len(df)} rows). This is the target length.")
        else:
            if len(df) != first_len:
                print(f"  -> CRITICAL ERROR: Length mismatch! {key} has {len(df)} rows, expected {first_len}.")
                return
            else:
                print(f"  -> {key} loaded (length match).")

        # Header assertions
        if key == "Sequence":
            if not expected_seq_columns.issubset(set(df.columns)):
                print(f"  -> CRITICAL ERROR: Sequence columns must include {expected_seq_columns}. Found: {list(df.columns)}")
                return
            # Print label counts before shuffle
            try:
                print("  -> Label counts (pre-shuffle):", df["Label"].value_counts().to_dict())
            except Exception:
                print("  -> WARNING: Could not compute pre-shuffle label counts.")
        else:
            # Shape headers should be 1..147
            if list(df.columns) != expected_shape_headers:
                print(f"  -> CRITICAL ERROR: {key} headers should be 1..147. Found: {list(df.columns)[:5]} ...")
                return

    # 2. Generate Random Index
    print(f"\nGenerating random permutation with seed {RANDOM_SEED}...")
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled_indices = rng.permutation(first_len)
    # Save permutation for reproducibility audits
    perm_path = os.path.join(BASE_DIR, "shuffle_indices.csv")
    pd.DataFrame({"index": shuffled_indices}).to_csv(perm_path, index=False)
    print(f"  -> Saved permutation to: {perm_path}")

    # 3. Apply Shuffle & Save
    print("\nShuffling and Saving...")
    for key, df in dfs.items():
        # Reorder the dataframe using the shuffled indices
        df_shuffled = df.iloc[shuffled_indices]
        
        # Save to the new path
        out_path = OUTPUT_FILES[key]
        
        # Create directory if it doesn't exist (safety check)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        df_shuffled.to_csv(out_path, index=False)
        print(f"  -> Saved {key} to: {out_path}")

        # Post-shuffle label counts (for Sequence file)
        if key == "Sequence":
            try:
                print("  -> Label counts (post-shuffle):", df_shuffled["Label"].value_counts().to_dict())
            except Exception:
                print("  -> WARNING: Could not compute post-shuffle label counts.")

    print("\nSUCCESS: All files shuffled synchronously!")

if __name__ == "__main__":
    main()