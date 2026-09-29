import os
import sys
import torch
import numpy as np
import pennylane as qml
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config
from src.data import load_and_split_data, create_dataloaders
from src.model import HybridQuantumClassifier, _quantum_circuit_shots_base

# We want to run a small sample (e.g. 100 rows) through the IonQ API so we don't blow through credits or timeout.
SAMPLE_SIZE = 100

class IonQClassifier(torch.nn.Module):
    def __init__(self, backend, shots=1000):
        super().__init__()
        self.fc_in = torch.nn.Linear(config.INPUT_DIM, config.N_QUBITS)
        
        # Instantiate IonQ device
        dev = qml.device(backend, wires=config.N_QUBITS, shots=shots)
        
        @qml.qnode(dev, interface="torch")
        def qc(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(config.N_QUBITS), rotation='X')
            qml.BasicEntanglerLayers(weights, wires=range(config.N_QUBITS), rotation=qml.RX)
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(config.N_QUBITS)]
            
        weight_shapes = {"weights": (config.N_QLAYERS, config.N_QUBITS)}
        self.qlayer = qml.qnn.TorchLayer(qc, weight_shapes)
        self.fc_out = torch.nn.Linear(config.N_QUBITS, 1)

    def forward(self, x):
        x = x.float()
        x_in = torch.sigmoid(self.fc_in(x)) * (np.pi / 2.0)
        q_out = self.qlayer(x_in)
        return self.fc_out(q_out)

def evaluate_on_device(backend_name, all_inputs, state_dict, device, shots=1000):
    start_time = time.time()
    try:
        model = IonQClassifier(backend=backend_name, shots=shots)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        probs = []
        with torch.no_grad():
            for i in range(0, len(all_inputs), config.BATCH_SIZE):
                batch = all_inputs[i:i+config.BATCH_SIZE]
                out = model(batch)
                probs.append(torch.sigmoid(out).cpu().numpy())
        probs = np.concatenate(probs).flatten()
        end_time = time.time()
        
        return {
            "status": "SUCCESS",
            "backend": backend_name,
            "shots": shots,
            "completion_time_seconds": end_time - start_time,
            "predictions_sample": probs[:5].tolist(),
            "mean_prediction": float(np.mean(probs))
        }
    except Exception as e:
        return {
            "status": "FAIL",
            "backend": backend_name,
            "error": str(e)
        }

def run_ionq_simulations():
    api_key = os.environ.get("IONQ_API_KEY")
    results = {
        "local_ideal": {},
        "ionq_ideal_simulator": {},
        "ionq_noise_model": {},
        "api_key_present": api_key is not None
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using compute device: {device}")
    
    train_df, test_df, _ = load_and_split_data()
    train_loader, test_loader = create_dataloaders(train_df, test_df)
    state_dict = torch.load("checkpoints/canonical_109param_weights.pth", map_location=device)
    
    # Grab a small sample of test inputs
    all_inputs = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            all_inputs.append(batch_x.to(device))
            if sum(x.shape[0] for x in all_inputs) >= SAMPLE_SIZE:
                break
    all_inputs = torch.cat(all_inputs)[:SAMPLE_SIZE]
    
    print(f"Evaluating {SAMPLE_SIZE} samples...")
    
    # 1. LOCAL IDEAL (already done in main script, doing here for quick comparison)
    print("Running LOCAL IDEAL simulator...")
    results["local_ideal"] = evaluate_on_device("default.qubit", all_inputs, state_dict, device, shots=1000)
    
    if api_key:
        print("IONQ_API_KEY found. Attempting IonQ remote simulations...")
        
        # 2. IONQ IDEAL SIMULATOR
        print("Running IONQ IDEAL SIMULATOR...")
        results["ionq_ideal_simulator"] = evaluate_on_device("ionq.simulator", all_inputs, state_dict, device, shots=1000)
        
        # 3. IONQ HARDWARE-NOISE SIMULATION
        print("Running IONQ NOISE-MODEL SIMULATOR (Aria-1)...")
        # For Pennylane-IonQ, noisy simulation can sometimes be requested by setting noise_model="aria-1"
        try:
            dev_noisy = qml.device("ionq.simulator", wires=config.N_QUBITS, shots=1000, noise_model="aria-1")
            
            # Inline the noisy model
            class NoisyIonQ(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc_in = torch.nn.Linear(config.INPUT_DIM, config.N_QUBITS)
                    @qml.qnode(dev_noisy, interface="torch")
                    def qc(inputs, weights):
                        qml.AngleEmbedding(inputs, wires=range(config.N_QUBITS), rotation='X')
                        qml.BasicEntanglerLayers(weights, wires=range(config.N_QUBITS), rotation=qml.RX)
                        return [qml.expval(qml.PauliZ(wires=i)) for i in range(config.N_QUBITS)]
                    weight_shapes = {"weights": (config.N_QLAYERS, config.N_QUBITS)}
                    self.qlayer = qml.qnn.TorchLayer(qc, weight_shapes)
                    self.fc_out = torch.nn.Linear(config.N_QUBITS, 1)

                def forward(self, x):
                    x = x.float()
                    x_in = torch.sigmoid(self.fc_in(x)) * (np.pi / 2.0)
                    q_out = self.qlayer(x_in)
                    return self.fc_out(q_out)
                    
            n_model = NoisyIonQ()
            n_model.load_state_dict(state_dict)
            n_model.to(device)
            n_model.eval()
            
            probs = []
            start_time = time.time()
            with torch.no_grad():
                for i in range(0, len(all_inputs), config.BATCH_SIZE):
                    batch = all_inputs[i:i+config.BATCH_SIZE]
                    out = n_model(batch)
                    probs.append(torch.sigmoid(out).cpu().numpy())
            probs = np.concatenate(probs).flatten()
            end_time = time.time()
            
            results["ionq_noise_model"] = {
                "status": "SUCCESS",
                "backend": "ionq.simulator (aria-1 noise)",
                "shots": 1000,
                "completion_time_seconds": end_time - start_time,
                "predictions_sample": probs[:5].tolist(),
                "mean_prediction": float(np.mean(probs))
            }
        except Exception as e:
            results["ionq_noise_model"] = {
                "status": "FAIL",
                "error": str(e)
            }
    else:
        print("IONQ_API_KEY not found. Skipping remote IonQ simulations.")
        results["ionq_ideal_simulator"] = {"status": "SKIPPED", "reason": "No API Key"}
        results["ionq_noise_model"] = {"status": "SKIPPED", "reason": "No API Key"}
        
    os.makedirs("results", exist_ok=True)
    with open("results/ionq_sim_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Simulation run complete. Results saved to results/ionq_sim_results.json")

if __name__ == "__main__":
    run_ionq_simulations()
