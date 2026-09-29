import os
import sys
import torch
import numpy as np
import pennylane as qml
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.data import load_and_split_data, create_dataloaders
from src.model import HybridQuantumClassifier, _quantum_circuit_shots_base

class DynamicShotClassifier(torch.nn.Module):
    def __init__(self, shots):
        super().__init__()
        self.fc_in = torch.nn.Linear(config.INPUT_DIM, config.N_QUBITS)
        weight_shapes = {"weights": (config.N_QLAYERS, config.N_QUBITS)}
        
        qc_shots = qml.set_shots(_quantum_circuit_shots_base, shots=shots)
        self.qlayer = qml.qnn.TorchLayer(qc_shots, weight_shapes)
        self.fc_out = torch.nn.Linear(config.N_QUBITS, 1)

    def forward(self, x):
        x = x.float()
        x_in = torch.sigmoid(self.fc_in(x)) * (np.pi / 2.0)
        q_out = self.qlayer(x_in)
        return self.fc_out(q_out)

def run_shot_stability():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading test data...")
    train_df, test_df, _ = load_and_split_data()
    train_loader, test_loader = create_dataloaders(train_df, test_df)
    
    print("Loading model and frozen weights...")
    model = HybridQuantumClassifier()
    state_dict = torch.load("checkpoints/canonical_109param_weights.pth", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    all_inputs = []
    all_labels = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            all_inputs.append(batch_x.to(device))
            all_labels.append(batch_y.numpy())
            
    all_inputs = torch.cat(all_inputs)
    all_labels = np.concatenate(all_labels)
    
    print("Running IDEAL inference (statevector)...")
    ideal_probs = []
    with torch.no_grad():
        for i in range(0, len(all_inputs), config.BATCH_SIZE):
            batch = all_inputs[i:i+config.BATCH_SIZE]
            out = model(batch)
            ideal_probs.append(torch.sigmoid(out).cpu().numpy())
    ideal_probs = np.concatenate(ideal_probs).flatten()
    
    ideal_roc = roc_auc_score(all_labels, ideal_probs)
    ideal_pr = average_precision_score(all_labels, ideal_probs)
    
    results = {
        "ideal": {
            "roc_auc": ideal_roc,
            "pr_auc": ideal_pr
        },
        "shot_based": {}
    }
    
    shot_counts = [100, 500, 1000, 5000]
    
    for shots in shot_counts:
        print(f"Running {shots}-shot inference...")
        shot_model = DynamicShotClassifier(shots)
        shot_model.load_state_dict(state_dict)
        shot_model.to(device)
        shot_model.eval()
        
        shot_probs = []
        with torch.no_grad():
            for i in range(0, len(all_inputs), config.BATCH_SIZE):
                batch = all_inputs[i:i+config.BATCH_SIZE]
                out = shot_model(batch)
                shot_probs.append(torch.sigmoid(out).cpu().numpy())
        shot_probs = np.concatenate(shot_probs).flatten()
        
        roc = roc_auc_score(all_labels, shot_probs)
        pr = average_precision_score(all_labels, shot_probs)
        p_corr, _ = pearsonr(ideal_probs, shot_probs)
        s_corr, _ = spearmanr(ideal_probs, shot_probs)
        mae = mean_absolute_error(ideal_probs, shot_probs)
        rmse = np.sqrt(mean_squared_error(ideal_probs, shot_probs))
        
        results["shot_based"][str(shots)] = {
            "roc_auc": float(roc),
            "pr_auc": float(pr),
            "pearson_corr": float(p_corr),
            "spearman_corr": float(s_corr),
            "mae": float(mae),
            "rmse": float(rmse)
        }
        
    os.makedirs("results", exist_ok=True)
    with open("results/shot_stability.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Shot stability analysis complete. Results saved to results/shot_stability.json")

if __name__ == "__main__":
    run_shot_stability()
