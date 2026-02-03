import numpy as np


def one_hot(seq, k):
    base_map = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1],
        'N': [0, 0, 0, 0]
    }

    seq = seq.upper()
    seq_len = len(seq)
    num_kmers = seq_len - k + 1

    code = np.empty(shape=(num_kmers, 4 * k))

    for i in range(num_kmers):
        kmer = seq[i:i + k]
        code[i] = np.concatenate([base_map.get(base, [0, 0, 0, 0]) for base in kmer])

    return code
