"""
config.py — Central configuration for reproducible quantum CRISPR experiments.

All hyperparameters, paths, and seeds are defined here so that every
experiment script draws from a single source of truth.
"""

import os
import sys

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
CHECKPOINTS_DIR = os.path.join(PROJECT_ROOT, "checkpoints")

DATASET_CSV = os.path.join(DATA_DIR, "Listgarten_22gRNA_wholeDataset.csv")

# ── Reproducibility ─────────────────────────────────────────────────────────
SEED = 42

# ── Dataset ──────────────────────────────────────────────────────────────────
INPUT_DIM = 24          # 23 bp + 1 zero-pad (matches original notebook)
TEST_SIZE = 0.2         # 80/20 stratified split (matches original notebook)

# ── Model architecture ───────────────────────────────────────────────────────
N_QUBITS = 4
N_QLAYERS = 1           # single variational layer
BOTTLENECK_OUT = N_QUBITS  # Linear(24 → 4)

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE = 32
LEARNING_RATE = 0.005
TRAINING_BATCHES = 500  # matches original notebook (500 batches × 32 = 16 000 examples)
POS_WEIGHT = 6816.11    # matches original notebook

# ── Shot-based simulation ────────────────────────────────────────────────────
N_SHOTS = 1000

# ── Checkpoint filenames ─────────────────────────────────────────────────────
CANONICAL_WEIGHTS = os.path.join(CHECKPOINTS_DIR, "canonical_109param_weights.pth")
TRAIN_INDICES = os.path.join(CHECKPOINTS_DIR, "train_indices.npy")
TEST_INDICES = os.path.join(CHECKPOINTS_DIR, "test_indices.npy")
HYPERPARAMS_JSON = os.path.join(CHECKPOINTS_DIR, "hyperparameters.json")


def get_versions() -> dict:
    """Return a dict of all relevant library versions."""
    import pennylane as qml
    import torch
    import numpy as np
    import pandas as pd
    import sklearn
    return {
        "python": sys.version,
        "pennylane": qml.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
    }


def ensure_dirs():
    """Create output directories if they don't exist."""
    for d in [DATA_DIR, RESULTS_DIR, PLOTS_DIR, CHECKPOINTS_DIR]:
        os.makedirs(d, exist_ok=True)
