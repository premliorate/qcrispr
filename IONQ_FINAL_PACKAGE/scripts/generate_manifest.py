import json
import hashlib
import os
import pennylane as qml
import torch
import sklearn
import numpy as np
import datetime
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

def get_file_hash(filepath):
    if not os.path.exists(filepath):
        return None
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Checkpoint
    ckpt_path = os.path.join(root, "checkpoints", "canonical_109param_weights.pth")
    ckpt_hash = get_file_hash(ckpt_path)
    
    # Dataset
    data_path = config.DATASET_CSV
    data_hash = get_file_hash(data_path)
    
    # Indices
    train_idx = os.path.join(root, "checkpoints", "train_indices.npy")
    test_idx = os.path.join(root, "checkpoints", "test_indices.npy")
    train_hash = get_file_hash(train_idx)
    test_hash = get_file_hash(test_idx)
    
    # Read parameters
    state = torch.load(ckpt_path, map_location='cpu')
    param_dict = {k: v.numpy().tolist() for k, v in state.items()}
    
    manifest = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "software_versions": {
            "python": sys.version,
            "torch": torch.__version__,
            "pennylane": qml.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__
        },
        "file_hashes_sha256": {
            "dataset": data_hash,
            "checkpoint": ckpt_hash,
            "train_indices": train_hash,
            "test_indices": test_hash
        },
        "hyperparameters": {
            "seed": config.SEED,
            "test_size": config.TEST_SIZE,
            "input_dim": config.INPUT_DIM,
            "n_qubits": config.N_QUBITS,
        },
        "preprocessing": {
            "method": "binary_mismatch",
            "match_value": 0.0,
            "mismatch_value": np.pi,
            "zero_pad_target": 24
        },
        "circuit_definition": {
            "qubits": 4,
            "embedding": "AngleEmbedding(rotation='X')",
            "variational": "BasicEntanglerLayers(rotation='X')",
            "entanglement": "CNOT ring (0->1, 1->2, 2->3, 3->0)",
            "measurement": "PauliZ expectation on all qubits"
        },
        "ionq_execution_plan": {
            "targets": ["ionq.simulator", "ionq.qpu"],
            "shots": [1000],
            "noise_model": "aria-1"
        },
        "frozen_model_parameters": param_dict
    }
    
    out_path = os.path.join(root, "results", "reproducibility_manifest.json")
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"Manifest written to {out_path}")

if __name__ == "__main__":
    main()
