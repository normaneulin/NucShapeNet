import pandas as pd
import numpy as np
from Bio import SeqIO
import os

# ==========================================
# CONFIGURATION
# ==========================================
# Counts for D. melanogaster
NUM_POS = 2900  # First 2900 are Nucleosomes
NUM_NEG = 2850  # Next 2850 are Linkers
TOTAL_ROWS = NUM_POS + NUM_NEG

# Base paths (resolved from this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_SEQ_DIR = os.path.join(SCRIPT_DIR, "Datasets", "Raw Data", "Sequence")
MASTER_SEQ_DIR = os.path.join(SCRIPT_DIR, "Datasets", "Master Data", "Sequence")

INPUT_FILE = "nucleosomes_vs_linkers_melanogaster.fas"
OUTPUT_FILE = "master_sequences.csv"

def process_sequences():
    os.makedirs(MASTER_SEQ_DIR, exist_ok=True)
    input_path = os.path.join(RAW_SEQ_DIR, INPUT_FILE)
    output_path = os.path.join(MASTER_SEQ_DIR, OUTPUT_FILE)
    print(f"--- Loading Sequences from {input_path} ---")
    
    ids = []
    sequences = []
    
    # Read the FASTA file
    try:
        for record in SeqIO.parse(input_path, "fasta"):
            # record.id captures the text after '>' up to the first space
            ids.append(record.id)
            # record.seq captures the sequence (stitched together)
            sequences.append(str(record.seq).upper())
            
    except FileNotFoundError:
        print(f"ERROR: File {input_path} not found. Please check the path.")
        return

    # Check if the file count matches your expectation
    if len(sequences) != TOTAL_ROWS:
        print(f"WARNING: File contains {len(sequences)} sequences, but you defined {TOTAL_ROWS} in config.")
        print("Please update NUM_POS and NUM_NEG in the script.")
    
    # Generate Labels
    # [1, 1, 1...] followed by [0, 0, 0...]
    labels = np.concatenate([np.ones(NUM_POS, dtype=int), np.zeros(NUM_NEG, dtype=int)])
    
    # Create DataFrame with 3 columns
    df = pd.DataFrame({
        'ID': ids,
        'Sequence': sequences,
        'Label': labels
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"SUCCESS: Saved {output_path} with {len(df)} rows.")
    print("Columns are: ID, Sequence, Label")

if __name__ == "__main__":
    process_sequences()