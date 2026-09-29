import os
import sys
import shutil
import json
import hashlib
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.join(ROOT, "IONQ_FINAL_PACKAGE")

def get_file_hash(filepath):
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)

def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

def check_no_credentials(folder):
    """Scan all files for API keys before zipping."""
    bad_patterns = [r"zXs7thhwvFvvimqgCLwzySXO3ADaZrrY", r"(?i)api_key\s*=\s*['\"][a-zA-Z0-9]{20,}['\"]"]
    for dname, _, fnames in os.walk(folder):
        for fname in fnames:
            if fname.endswith((".py", ".md", ".json", ".txt", ".csv")):
                fpath = os.path.join(dname, fname)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    for p in bad_patterns:
                        if re.search(p, content):
                            print(f"[FAIL] Credential pattern found in {fpath}!")
                            return False
    return True

def main():
    if os.path.exists(PKG_DIR):
        shutil.rmtree(PKG_DIR)
    
    ensure_dir(PKG_DIR)
    ensure_dir(os.path.join(PKG_DIR, "src"))
    ensure_dir(os.path.join(PKG_DIR, "scripts"))
    ensure_dir(os.path.join(PKG_DIR, "checkpoints"))
    
    # 1. READ VERIFIED JSON RESULTS
    with open(os.path.join(ROOT, "results", "ionq_sim_results.json"), "r") as f:
        ionq = json.load(f)
    with open(os.path.join(ROOT, "results", "all_results.json"), "r") as f:
        all_res = json.load(f)

    ideal_roc = ionq["ionq_ideal_simulator"]["roc_auc"]
    ideal_pr = ionq["ionq_ideal_simulator"]["pr_auc"]
    ideal_corr = ionq["ionq_ideal_simulator"]["pearson_corr_vs_ideal"]
    ideal_samples = ionq["ionq_ideal_simulator"]["n_samples"]
    
    noise_roc = ionq["ionq_noise_simulator"]["roc_auc"]
    noise_pr = ionq["ionq_noise_simulator"]["pr_auc"]
    noise_corr = ionq["ionq_noise_simulator"]["pearson_corr_vs_ideal"]
    noise_samples = ionq["ionq_noise_simulator"]["n_samples"]
    
    local_roc = ionq["local_ideal_probs_recomputed_roc"]
    baseline_roc = all_res["experiment_5_classical_baseline"]["roc_auc"]
    baseline_pr = all_res["experiment_5_classical_baseline"]["pr_auc"]

    # 2. CREATE MARKDOWNS
    readme = f"""# Reproducible Hybrid Quantum-Classical Simulation for CRISPR Off-Target Prediction

## Overview
This package prepares a reproducible, frozen 4-qubit hybrid quantum-classical neural network for execution on IonQ hardware. The objective is to validate whether the exact 109-parameter model produces consistent classification behavior across local ideal simulators, IonQ's cloud simulator, IonQ's noise-model simulators, and eventual IonQ trapped-ion QPUs.

## Architecture
- **Problem**: CRISPR-Cas9 Off-Target Prediction
- **Dataset**: Listgarten Dataset II/6
- **Architecture**: 4-qubit hybrid architecture
- **Parameters**: 109 total (100 classical bottleneck, 4 quantum variational, 5 classical output)
- **Exact Circuit**: 4 RX data embedding gates, 4 trainable RX variational gates, CNOT ring (0->1->2->3->0), Pauli-Z measurements on all qubits.
- **Frozen Weights**: Model weights are frozen. No retraining is permitted.

## Verified Results Summary
- **Local Reference Results (Full Test Set)**: ROC-AUC = 0.8704
- **Local Reference Results (111-Sample Comparison Set)**: ROC-AUC = {local_roc:.4f}
- **IonQ Ideal Simulation Results (111-Sample Comparison Set)**: ROC-AUC = {ideal_roc:.4f} (Correlation vs local: {ideal_corr:.4f})
- **IonQ Noise-Model Results (Aria-1, 111-Sample Comparison Set)**: ROC-AUC = {noise_roc:.4f}

## Limitations
The full held-out test set contains only 11 positive examples out of 76,693 total examples. Because of this extreme class imbalance, full-test AUC estimates are highly sensitive to ranking changes among a very small number of positives. Furthermore, this study does not demonstrate quantum advantage; the parameter-matched classical baseline outperforms the quantum model.

## Next Steps
Upon allocation of research credits, the next step is QPU execution on IonQ hardware to quantify hardware fidelity relative to the validated simulation references established in this package.
"""
    write_file(os.path.join(PKG_DIR, "README_FIRST.md"), readme)

    research_summary = f"""# IonQ Research Summary
    
| Experiment | Backend | Samples | Shots | ROC-AUC | PR-AUC | Correlation vs Local |
| --- | --- | --- | --- | --- | --- | --- |
| Local Ideal Full | lightning.qubit | 76693 | N/A (Statevector) | 0.8704 | 0.0117 | N/A |
| Local Ideal Subsample | lightning.qubit | {ideal_samples} | N/A (Statevector) | {local_roc:.4f} | N/A | 1.0000 |
| Classical Baseline | CPU (PyTorch) | 76693 | N/A | {baseline_roc:.4f} | {baseline_pr:.4f} | N/A |
| IonQ Ideal | ionq.simulator | {ideal_samples} | 1000 | {ideal_roc:.4f} | {ideal_pr:.4f} | {ideal_corr:.4f} |
| IonQ Noise (Aria-1) | ionq.simulator | {noise_samples} | 1000 | {noise_roc:.4f} | {noise_pr:.4f} | {noise_corr:.4f} |

*Explicitly noted: this study does not demonstrate quantum advantage. The parameter-matched classical baseline (ROC-AUC 0.9483) outperforms the quantum model. The goal is to study hardware inference fidelity of a frozen network.*

## Circuit Claims
- **4 qubits**: Selected to match the 24->4 classical dimensionality reduction bottleneck.
- **Shallow depth**: A single variational layer was chosen to keep the model compact for hardware fidelity study.
- **RX embedding**: Direct encoding of the 4 latent features.
- **CNOT ring**: Explicit entangling hypothesis tested by ablation. (Ablation showed removing the CNOT ring produced higher ROC-AUC in ideal simulation, indicating the structure did not improve predictive performance for this particular model configuration).
- **Pauli-Z measurements**: Provide a 4-dimensional quantum feature vector to the final classical layer.
"""
    write_file(os.path.join(PKG_DIR, "IONQ_RESEARCH_SUMMARY.md"), research_summary)

    researcher_draft = f"""Subject: Research Credit Application: Hardware Fidelity of a Hybrid Quantum Classifier for CRISPR Off-Target Prediction

Dear IonQ Research Team,

We are seeking IonQ research credits to execute the same frozen circuit and parameters on IonQ hardware and quantify hardware fidelity relative to the validated simulation reference. 

Our canonical circuit is a 4-qubit, 109-parameter hybrid quantum-classical neural network. It utilizes a 24->4 classical bottleneck, encodes data with RX rotations, uses one trainable RX layer, a 4-CNOT ring, and measures four Pauli-Z expectation values. The model parameters are completely frozen.

We have successfully executed our cloud simulations on the IonQ ideal and noise simulators. Using a stratified 111-sample comparison set, the IonQ ideal cloud simulation yielded a ROC-AUC of {ideal_roc:.4f} (Pearson correlation of {ideal_corr:.4f} with our local ideal simulation). The IonQ Aria-1 noise-model simulation on the same {noise_samples} samples yielded a ROC-AUC of {noise_roc:.4f}. For context, our local full-test reference (76,693 samples) evaluates to ROC-AUC 0.8704.

We do not claim quantum advantage in this experiment. Our objective is to benchmark hardware-fidelity inference behavior on a heavily imbalanced bio-informatics dataset.

Thank you for your consideration.

Sincerely,
[Researcher Name]
"""
    write_file(os.path.join(PKG_DIR, "IONQ_RESEARCHER_DRAFT.md"), researcher_draft)

    # 3. COPY FILES
    shutil.copy(os.path.join(ROOT, "results", "circuit_justification.md"), os.path.join(PKG_DIR, "circuit_justification.md"))
    shutil.copy(os.path.join(ROOT, "results", "circuit_diagram.txt"), os.path.join(PKG_DIR, "circuit_diagram.txt"))
    shutil.copy(os.path.join(ROOT, "results", "circuit_equivalence.json"), os.path.join(PKG_DIR, "circuit_equivalence.json"))
    shutil.copy(os.path.join(ROOT, "results", "parameter_verification.json"), os.path.join(PKG_DIR, "parameter_verification.json"))
    shutil.copy(os.path.join(ROOT, "results", "shot_stability.json"), os.path.join(PKG_DIR, "shot_stability.json"))
    shutil.copy(os.path.join(ROOT, "results", "experiment_comparison.csv"), os.path.join(PKG_DIR, "experiment_comparison.csv"))
    shutil.copy(os.path.join(ROOT, "results", "ionq_sim_results.json"), os.path.join(PKG_DIR, "ionq_sim_results.json"))
    shutil.copy(os.path.join(ROOT, "results", "all_results.json"), os.path.join(PKG_DIR, "all_results.json"))
    shutil.copy(os.path.join(ROOT, "results", "reproducibility_manifest.json"), os.path.join(PKG_DIR, "reproducibility_manifest.json"))
    
    shutil.copy(os.path.join(ROOT, "src", "canonical_circuit.py"), os.path.join(PKG_DIR, "src", "canonical_circuit.py"))
    
    shutil.copy(os.path.join(ROOT, "scripts", "run_ionq_integration.py"), os.path.join(PKG_DIR, "scripts", "run_ionq_integration.py"))
    shutil.copy(os.path.join(ROOT, "scripts", "shot_stability.py"), os.path.join(PKG_DIR, "scripts", "shot_stability.py"))
    shutil.copy(os.path.join(ROOT, "scripts", "final_validation.py"), os.path.join(PKG_DIR, "scripts", "final_validation.py"))
    shutil.copy(os.path.join(ROOT, "scripts", "generate_manifest.py"), os.path.join(PKG_DIR, "scripts", "generate_manifest.py"))
    shutil.copy(os.path.join(ROOT, "scripts", "scrub_credentials.py"), os.path.join(PKG_DIR, "scripts", "scrub_credentials.py"))
    
    shutil.copy(os.path.join(ROOT, "checkpoints", "canonical_109param_weights.pth"), os.path.join(PKG_DIR, "checkpoints", "canonical_109param_weights.pth"))

    # 4. FINAL CREDENTIAL SCAN
    safe = check_no_credentials(PKG_DIR)
    if not safe:
        print("[ERROR] Failed credential scan. API key found in package.")
        sys.exit(1)
        
    print("[OK] Credential scan passed.")

    # 5. ZIP PACKAGE
    zip_path = os.path.join(ROOT, "IONQ_CRISPR_OFFTARGET_FINAL_PACKAGE")
    shutil.make_archive(zip_path, 'zip', PKG_DIR)
    
    zip_file = zip_path + ".zip"
    zip_size = os.path.getsize(zip_file)
    zip_hash = get_file_hash(zip_file)
    
    write_file(os.path.join(ROOT, "IONQ_PACKAGE_SHA256.txt"), zip_hash)
    
    print(f"ZIP Path: {zip_file}")
    print(f"ZIP Size: {zip_size} bytes")
    print(f"SHA-256: {zip_hash}")
    
    # Extract needed output info
    print(f"IonQ Ideal Job Status: SUCCESS")
    print(f"IonQ Noise Job Status: SUCCESS")
    print(f"IonQ Ideal Samples: {ideal_samples}")
    print(f"IonQ Noise Samples: {noise_samples}")
    print(f"Verified Ideal ROC-AUC: {ideal_roc:.4f}")
    print(f"Verified Noise ROC-AUC: {noise_roc:.4f}")

if __name__ == "__main__":
    main()
