import pandas as pd
import numpy as np
import os

# ==========================================
# CONFIGURATION
# ==========================================
SEQ_LEN = 147

# Base paths (resolved from this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_SHAPE_DIR = os.path.join(SCRIPT_DIR, "Datasets", "Raw Data", "Shape")
MASTER_SHAPE_DIR = os.path.join(SCRIPT_DIR, "Datasets", "Master Data", "Shape")

# Map your specific file names to the Feature Name
SHAPE_FILES = {
    "MGW":  "nucleosomes_vs_linkers_melanogaster.fas.MGW",
    "ProT": "nucleosomes_vs_linkers_melanogaster.fas.ProT",
    "Roll": "nucleosomes_vs_linkers_melanogaster.fas.Roll",
    "HelT": "nucleosomes_vs_linkers_melanogaster.fas.HelT",
    "EP":   "nucleosomes_vs_linkers_melanogaster.fas.EP"
}

# Define which features are "Step Parameters" (Values between bases)
STEP_FEATURES = ["Roll", "HelT"]

def parse_shape_file(filename):
    """
    Custom parser for DNAShapeR output that is split across multiple lines.
    Returns a list of lists (one list of floats per sequence).
    """
    records = []
    current_values = []
    
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # If we hit a header (starts with >)
            if line.startswith(">"):
                # If we have collected values for the PREVIOUS sequence, save them
                if current_values:
                    records.append(current_values)
                    current_values = []
                # (We ignore the header text itself)
                continue
            
            # This is a data line (e.g., "NA,NA,-1.93...")
            # Split by comma
            parts = line.split(',')
            
            # Process each value
            for p in parts:
                p = p.strip()
                if p == "NA" or p == "":
                    val = 0.0
                else:
                    try:
                        val = float(p)
                    except ValueError:
                        val = 0.0
                current_values.append(val)

        # Don't forget to save the LAST sequence in the file
        if current_values:
            records.append(current_values)
            
    return records

def process_shapes():
    print("--- Processing Shape Data ---")
    os.makedirs(MASTER_SHAPE_DIR, exist_ok=True)
    
    # Generate column names "1", "2", ... "147"
    headers = [str(i) for i in range(1, SEQ_LEN + 1)]

    for feature_name, filename in SHAPE_FILES.items():
        print(f"\nProcessing {feature_name}...")

        input_path = os.path.join(RAW_SHAPE_DIR, filename)
        if not os.path.exists(input_path):
            print(f"ERROR: Could not find {input_path}. Skipping.")
            continue

        # 1. USE CUSTOM PARSER instead of read_csv
        try:
            data_list = parse_shape_file(input_path)
            # Convert to DataFrame
            df = pd.DataFrame(data_list)
        except Exception as e:
            print(f"ERROR reading {filename}: {e}")
            continue
        
        print(f"  -> Parsed {len(df)} sequences.")
        
        # 2. Check Dimensions & Pad
        current_cols = df.shape[1]
        
        # LOGIC FOR STANDARD FEATURES (MGW, ProT, EP) -> Should be 147
        if feature_name not in STEP_FEATURES:
            if current_cols == SEQ_LEN:
                print(f"  -> Dimensions OK ({current_cols}). Assigning headers.")
                df.columns = headers
            else:
                print(f"  -> WARNING: {feature_name} should be {SEQ_LEN} but is {current_cols}!")
                # Force resize
                df_fixed = pd.DataFrame(0.0, index=df.index, columns=headers)
                limit = min(current_cols, SEQ_LEN)
                df_fixed.iloc[:, :limit] = df.iloc[:, :limit].values
                df = df_fixed

        # LOGIC FOR STEP FEATURES (Roll, HelT) -> Should be 146, need padding to 147
        else:
            if current_cols == SEQ_LEN - 1:
                print(f"  -> Step Feature identified ({current_cols} cols). Padding END with 0.0.")
                # Pad at the end (effectively creating column "147")
                df[df.shape[1]] = 0.0  
                df.columns = headers
            elif current_cols == SEQ_LEN:
                 print(f"  -> Note: {feature_name} already has {SEQ_LEN} cols. Assuming already padded.")
                 df.columns = headers
            else:
                print(f"  -> WARNING: {feature_name} has odd dimensions {current_cols}. Forcing resize.")
                df_fixed = pd.DataFrame(0.0, index=df.index, columns=headers)
                limit = min(current_cols, SEQ_LEN)
                df_fixed.iloc[:, :limit] = df.iloc[:, :limit].values
                df = df_fixed

        # 3. Save to Master CSV
        output_name = f"master_{feature_name}.csv"
        output_path = os.path.join(MASTER_SHAPE_DIR, output_name)
        df.to_csv(output_path, index=False)
        print(f"  -> SUCCESS: Saved {output_path}")

if __name__ == "__main__":
    process_shapes()