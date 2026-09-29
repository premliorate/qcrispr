"""
data.py — Dataset loading, preprocessing, and PyTorch Dataset/DataLoader creation.

Reproduces the exact preprocessing from the original notebook:
  - Binary mismatch encoding: match → 0.0, mismatch → π
  - Zero-pad to 24 features (23 bp + 1 pad)
  - Stratified 80/20 train/test split with random_state=42
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from . import config


def encode_dna_to_angles(sgRNA: str, target_dna: str, target_length: int = 24) -> np.ndarray:
    """
    Binary mismatch encoding — exact reproduction of the original notebook.

    For each position i in [0, min(len(sgRNA), len(target_dna))):
        match   → 0.0
        mismatch → π

    Pad/truncate to `target_length` (default 24).
    """
    sgRNA = str(sgRNA).strip().upper()
    target_dna = str(target_dna).strip().upper()
    compare_len = min(len(sgRNA), len(target_dna))

    angles = []
    for i in range(compare_len):
        if sgRNA[i] == target_dna[i]:
            angles.append(0.0)
        else:
            angles.append(np.pi)

    # Pad to target_length with zeros (matches original notebook)
    while len(angles) < target_length:
        angles.append(0.0)

    # Truncate if somehow longer
    angles = angles[:target_length]

    return np.array(angles, dtype=np.float32)


class CRISPRDataset(Dataset):
    """PyTorch Dataset for CRISPR off-target prediction."""

    def __init__(self, dataframe: pd.DataFrame):
        self.data = dataframe.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        sgRNA = row["on_seq"]
        target_dna = row["off_seq"]
        encoded = encode_dna_to_angles(sgRNA, target_dna)
        x = torch.tensor(encoded, dtype=torch.float32)
        y = torch.tensor([row["Label"]], dtype=torch.float32)
        return x, y


def load_and_split_data():
    """
    Load the Listgarten Dataset II/6 and reproduce the original split.

    Returns:
        train_df, test_df, dataset_stats (dict)
    """
    df = pd.read_csv(config.DATASET_CSV)

    # Standardize column names (match original notebook)
    col_map = {}
    for c in df.columns:
        cl = c.lower()
        if "on" in cl or "sg" in cl:
            col_map[c] = "on_seq"
        elif "off" in cl:
            col_map[c] = "off_seq"
        elif "lab" in cl:
            col_map[c] = "Label"
    df = df.rename(columns=col_map)

    total_rows = len(df)
    n_positive = int(df["Label"].sum())
    n_negative = total_rows - n_positive

    # Check sequence lengths
    seq_lengths = df["on_seq"].str.len().unique()

    # Stratified split — exact same as original notebook
    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=df["Label"]
    )

    train_pos = int(train_df["Label"].sum())
    train_neg = len(train_df) - train_pos
    test_pos = int(test_df["Label"].sum())
    test_neg = len(test_df) - test_pos

    stats = {
        "total_rows": total_rows,
        "total_positive": n_positive,
        "total_negative": n_negative,
        "class_imbalance_ratio": f"1:{n_negative // n_positive}" if n_positive > 0 else "N/A",
        "sequence_lengths_found": sorted(seq_lengths.tolist()),
        "input_dimension": config.INPUT_DIM,
        "train_rows": len(train_df),
        "train_positive": train_pos,
        "train_negative": train_neg,
        "test_rows": len(test_df),
        "test_positive": test_pos,
        "test_negative": test_neg,
    }

    return train_df, test_df, stats


def create_dataloaders(train_df, test_df):
    """Create PyTorch DataLoaders for train and test sets."""
    train_dataset = CRISPRDataset(train_df)
    test_dataset = CRISPRDataset(test_df)

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(config.SEED)
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False
    )

    return train_loader, test_loader
