"""
run_experiments.py — Master experiment runner for reproducible quantum CRISPR simulation.

Executes all P0 and P1 experiments:
  1. Train canonical 109-param model (ideal statevector)
  2. Evaluate on full test set (ideal simulation)
  3. Shot-based simulation (1000 shots, frozen weights)
  4. Entanglement ablation (remove CNOTs)
  5. Embedding ablation (bounded vs unbounded)
  6. Parameter-matched classical baseline (109 params)

Outputs: results JSON, plots, confusion matrices, circuit diagram.
"""

import json
import os
import sys
import time
import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report,
    f1_score, matthews_corrcoef
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config
from src.data import load_and_split_data, create_dataloaders
from src.model import (
    HybridQuantumClassifier,
    HybridQuantumClassifierShots,
    ClassicalBaseline109,
    draw_circuit,
)


def set_seed(seed=config.SEED):
    """Set all random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_model(model, train_loader, model_name="model", max_batches=config.TRAINING_BATCHES):
    """
    Train a model using BCEWithLogitsLoss with pos_weight.

    Reproduces the original notebook's training procedure:
      - Adam optimizer, lr=0.005
      - BCEWithLogitsLoss(pos_weight=6816.11)
      - 500 batches (not full epochs)
    """
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([config.POS_WEIGHT]))
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    model.train()
    losses = []
    t0 = time.time()

    for i, (batch_X, batch_y) in enumerate(train_loader):
        if i >= max_batches:
            break

        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        labels = batch_y.squeeze().float()

        # Handle single-item batches
        if outputs.dim() == 0:
            outputs = outputs.unsqueeze(0)
            labels = labels.unsqueeze(0)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  [{model_name}] Batch {i+1}/{max_batches} | "
                  f"Loss: {loss.item():.4f} | Time: {elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"  [{model_name}] Training complete: {len(losses)} batches in {elapsed:.1f}s")
    return losses


def evaluate_model(model, test_loader, model_name="model", threshold=0.5):
    """
    Evaluate a model on the FULL test set.

    Returns a dict with all metrics. Does NOT tune threshold on test set.
    Uses fixed threshold=0.5 for confusion matrix (honest default).
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X).squeeze()
            labels = batch_y.squeeze().float()

            if outputs.dim() == 0:
                outputs = outputs.unsqueeze(0)
                labels = labels.unsqueeze(0)

            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs)
            all_preds.extend(probs.numpy().tolist())
            all_labels.extend(labels.numpy().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    n_total = len(all_labels)
    n_positive = int(all_labels.sum())
    n_negative = n_total - n_positive

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(all_labels, all_preds)
    except ValueError:
        roc_auc = float("nan")

    # PR-AUC
    try:
        pr_auc = average_precision_score(all_labels, all_preds)
    except ValueError:
        pr_auc = float("nan")

    # Confusion matrix at fixed threshold (no test-set tuning!)
    pred_binary = (all_preds >= threshold).astype(int)
    cm = confusion_matrix(all_labels, pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    try:
        f1 = f1_score(all_labels, pred_binary)
    except:
        f1 = 0.0

    try:
        mcc = matthews_corrcoef(all_labels, pred_binary)
    except:
        mcc = 0.0

    results = {
        "model_name": model_name,
        "n_test_total": n_total,
        "n_test_positive": n_positive,
        "n_test_negative": n_negative,
        "roc_auc": round(roc_auc, 6),
        "pr_auc": round(pr_auc, 6),
        "threshold_used": threshold,
        "threshold_note": "Fixed at 0.5 — NOT tuned on test set",
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "sensitivity_recall": round(sensitivity, 6),
        "specificity": round(specificity, 6),
        "precision": round(precision_val, 6),
        "f1_score": round(f1, 6),
        "mcc": round(mcc, 6),
    }

    print(f"\n  [{model_name}] Evaluation on {n_total} test samples "
          f"({n_positive} pos, {n_negative} neg):")
    print(f"    ROC-AUC     = {roc_auc:.4f}")
    print(f"    PR-AUC      = {pr_auc:.4f}")
    print(f"    Sensitivity = {sensitivity:.4f}")
    print(f"    Specificity = {specificity:.4f}")
    print(f"    F1          = {f1:.4f}")
    print(f"    MCC         = {mcc:.4f}")
    print(f"    Confusion   = TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    return results, all_preds, all_labels


def plot_roc_curves(results_dict, filename="roc_comparison.png"):
    """Plot ROC curves for all experiments."""
    plt.figure(figsize=(10, 8))

    colors = {
        "Quantum (ideal)": "#1f77b4",
        "Quantum (1000 shots)": "#2ca02c",
        "No entanglement": "#ff7f0e",
        "Unbounded embedding": "#9467bd",
        "Classical baseline": "#d62728",
    }

    for name, data in results_dict.items():
        preds, labels = data["preds"], data["labels"]
        fpr, tpr, _ = roc_curve(labels, preds)
        auc_val = auc(fpr, tpr)
        color = colors.get(name, "#333333")
        plt.plot(fpr, tpr, lw=2.5, color=color,
                 label=f"{name} (AUC={auc_val:.4f})")

    plt.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Positive Rate", fontsize=13)
    plt.ylabel("True Positive Rate (Sensitivity)", fontsize=13)
    plt.title("ROC Comparison: Hybrid Quantum Classifier Experiments", fontsize=14, fontweight="bold")
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ROC plot saved: {filename}")


def plot_pr_curves(results_dict, filename="pr_comparison.png"):
    """Plot Precision-Recall curves for all experiments."""
    plt.figure(figsize=(10, 8))

    colors = {
        "Quantum (ideal)": "#1f77b4",
        "Quantum (1000 shots)": "#2ca02c",
        "No entanglement": "#ff7f0e",
        "Unbounded embedding": "#9467bd",
        "Classical baseline": "#d62728",
    }

    for name, data in results_dict.items():
        preds, labels = data["preds"], data["labels"]
        precision_arr, recall_arr, _ = precision_recall_curve(labels, preds)
        ap = average_precision_score(labels, preds)
        color = colors.get(name, "#333333")
        plt.plot(recall_arr, precision_arr, lw=2.5, color=color,
                 label=f"{name} (AP={ap:.4f})")

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("Recall", fontsize=13)
    plt.ylabel("Precision", fontsize=13)
    plt.title("Precision-Recall Comparison", fontsize=14, fontweight="bold")
    plt.legend(loc="upper right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, filename), dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  PR plot saved: {filename}")


def main():
    print("=" * 70)
    print("REPRODUCIBLE QUANTUM CRISPR SIMULATION PACKAGE")
    print("4-Qubit Hybrid Variational Quantum Classifier")
    print(f"Started: {datetime.datetime.now().isoformat()}")
    print("=" * 70)

    config.ensure_dirs()
    set_seed()

    # ── Library versions ─────────────────────────────────────────────────
    versions = config.get_versions()
    print("\nLibrary Versions:")
    for k, v in versions.items():
        print(f"  {k}: {v}")

    # ── Step 1: Load and inspect dataset ─────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: Dataset Loading and Inspection")
    print("=" * 70)

    train_df, test_df, dataset_stats = load_and_split_data()

    print(f"\nDataset: Listgarten_22gRNA_wholeDataset.csv")
    print(f"  Total rows:         {dataset_stats['total_rows']}")
    print(f"  Positive (off-target): {dataset_stats['total_positive']}")
    print(f"  Negative (safe):    {dataset_stats['total_negative']}")
    print(f"  Class imbalance:    {dataset_stats['class_imbalance_ratio']}")
    print(f"  Sequence lengths:   {dataset_stats['sequence_lengths_found']}")
    print(f"  Input dimension:    {dataset_stats['input_dimension']} (23bp + 1 zero-pad)")
    print(f"  Train: {dataset_stats['train_rows']} ({dataset_stats['train_positive']} pos, {dataset_stats['train_negative']} neg)")
    print(f"  Test:  {dataset_stats['test_rows']} ({dataset_stats['test_positive']} pos, {dataset_stats['test_negative']} neg)")

    # Save indices for reproducibility
    np.save(config.TRAIN_INDICES, train_df.index.values)
    np.save(config.TEST_INDICES, test_df.index.values)

    train_loader, test_loader = create_dataloaders(train_df, test_df)

    # ── Step 2: Circuit diagram ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 2: Circuit Diagram")
    print("=" * 70)

    circuit_text = draw_circuit()
    print("\nCanonical 4-qubit quantum circuit:")
    print(circuit_text)

    with open(os.path.join(config.RESULTS_DIR, "circuit_diagram.txt"), "w", encoding="utf-8") as f:
        f.write("Canonical 4-Qubit Hybrid Variational Quantum Classifier Circuit\n")
        f.write("=" * 60 + "\n\n")
        f.write("Architecture:\n")
        f.write("  Input: 24-dim binary mismatch vector\n")
        f.write("  Linear(24, 4) + sigmoid × (π/2) → bounded angles ∈ [0, π/2]\n")
        f.write("  RX(angle[i]) on qubit i — data embedding (4 qubits)\n")
        f.write("  RX(θ[i]) on qubit i — trainable variational layer (4 params)\n")
        f.write("  CNOT ring: 0→1, 1→2, 2→3, 3→0\n")
        f.write("  ⟨Z⟩ measurement on all 4 qubits\n")
        f.write("  Linear(4, 1) → logit output\n\n")
        f.write("Total trainable parameters: 109\n")
        f.write("  - fc_in:  100 (24×4 weights + 4 biases)\n")
        f.write("  - qlayer:   4 (1 layer × 4 qubits)\n")
        f.write("  - fc_out:   5 (4×1 weights + 1 bias)\n\n")
        f.write("PennyLane circuit drawing:\n")
        f.write(circuit_text + "\n")

    # ── Step 3: Train canonical model ────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 3: Train Canonical 109-Parameter Model (Ideal Simulation)")
    print("=" * 70)

    set_seed()
    model = HybridQuantumClassifier(use_entanglement=True, bounded_embedding=True)
    param_info = model.count_parameters()
    print(f"\nParameter breakdown:")
    for k, v in param_info.items():
        print(f"  {k}: {v}")
    assert param_info["total"] == 109, f"Expected 109 params, got {param_info['total']}"

    print(f"\nTraining with: lr={config.LEARNING_RATE}, batch_size={config.BATCH_SIZE}, "
          f"pos_weight={config.POS_WEIGHT}, max_batches={config.TRAINING_BATCHES}")

    train_losses = train_model(model, train_loader, model_name="Canonical-109")

    # Save weights
    torch.save(model.state_dict(), config.CANONICAL_WEIGHTS)
    print(f"  Weights saved: {config.CANONICAL_WEIGHTS}")

    # Save hyperparameters
    hparams = {
        "seed": config.SEED,
        "input_dim": config.INPUT_DIM,
        "n_qubits": config.N_QUBITS,
        "n_qlayers": config.N_QLAYERS,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "training_batches": config.TRAINING_BATCHES,
        "pos_weight": config.POS_WEIGHT,
        "test_size": config.TEST_SIZE,
        "total_parameters": param_info["total"],
        "parameter_breakdown": param_info,
        "embedding": "RX",
        "variational": "RX",
        "entanglement": "CNOT ring (0→1, 1→2, 2→3, 3→0)",
        "bounded_embedding": "sigmoid(Wx+b) × π/2",
        "measurement": "⟨Z⟩ on all qubits",
        "loss": "BCEWithLogitsLoss(pos_weight=6816.11)",
        "optimizer": "Adam",
        "versions": versions,
    }
    with open(config.HYPERPARAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(hparams, f, indent=2)
    print(f"  Hyperparameters saved: {config.HYPERPARAMS_JSON}")

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: Ideal statevector simulation
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Ideal Statevector Simulation (Full Test Set)")
    print("=" * 70)

    results_ideal, preds_ideal, labels_ideal = evaluate_model(
        model, test_loader, model_name="Quantum (ideal)"
    )

    all_results = {"experiment_1_ideal": results_ideal}
    curve_data = {"Quantum (ideal)": {"preds": preds_ideal, "labels": labels_ideal}}

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: Shot-based simulation
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: Shot-Based Simulation ({config.N_SHOTS} shots)")
    print("=" * 70)

    # Create shot-based model and copy frozen weights
    model_shots = HybridQuantumClassifierShots()
    model_shots.load_state_dict(model.state_dict())

    results_shots, preds_shots, labels_shots = evaluate_model(
        model_shots, test_loader, model_name="Quantum (1000 shots)"
    )

    # Correlation between ideal and shot-based predictions
    correlation = np.corrcoef(preds_ideal, preds_shots)[0, 1]
    mae = np.mean(np.abs(preds_ideal - preds_shots))
    results_shots["correlation_with_ideal"] = round(correlation, 6)
    results_shots["mae_vs_ideal"] = round(mae, 6)
    print(f"  Correlation with ideal: {correlation:.4f}")
    print(f"  MAE vs ideal:           {mae:.6f}")

    all_results["experiment_2_shots"] = results_shots
    curve_data["Quantum (1000 shots)"] = {"preds": preds_shots, "labels": labels_shots}

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: Entanglement ablation
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Entanglement Ablation (Remove CNOT Gates)")
    print("=" * 70)

    set_seed()
    model_no_ent = HybridQuantumClassifier(use_entanglement=False, bounded_embedding=True)
    print(f"  Parameters: {model_no_ent.count_parameters()}")

    train_losses_no_ent = train_model(model_no_ent, train_loader, model_name="No-Entanglement")

    results_no_ent, preds_no_ent, labels_no_ent = evaluate_model(
        model_no_ent, test_loader, model_name="No entanglement"
    )
    all_results["experiment_3_no_entanglement"] = results_no_ent
    curve_data["No entanglement"] = {"preds": preds_no_ent, "labels": labels_no_ent}

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 4: Embedding ablation (bounded vs unbounded)
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Embedding Ablation (Unbounded/Raw Angles)")
    print("=" * 70)

    set_seed()
    model_unbounded = HybridQuantumClassifier(use_entanglement=True, bounded_embedding=False)
    print(f"  Parameters: {model_unbounded.count_parameters()}")

    train_losses_unb = train_model(model_unbounded, train_loader, model_name="Unbounded-Embedding")

    results_unbounded, preds_unbounded, labels_unbounded = evaluate_model(
        model_unbounded, test_loader, model_name="Unbounded embedding"
    )
    all_results["experiment_4_unbounded_embedding"] = results_unbounded
    curve_data["Unbounded embedding"] = {"preds": preds_unbounded, "labels": labels_unbounded}

    # ══════════════════════════════════════════════════════════════════════
    # EXPERIMENT 5: Parameter-matched classical baseline
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Parameter-Matched Classical Baseline (109 params)")
    print("=" * 70)

    set_seed()
    model_classical = ClassicalBaseline109()
    classical_params = sum(p.numel() for p in model_classical.parameters())
    print(f"  Classical baseline parameters: {classical_params}")

    train_losses_cl = train_model(model_classical, train_loader, model_name="Classical-109")

    results_classical, preds_classical, labels_classical = evaluate_model(
        model_classical, test_loader, model_name="Classical baseline"
    )
    all_results["experiment_5_classical_baseline"] = results_classical
    curve_data["Classical baseline"] = {"preds": preds_classical, "labels": labels_classical}

    # ══════════════════════════════════════════════════════════════════════
    # Generate plots and summary
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GENERATING PLOTS AND SUMMARY")
    print("=" * 70)

    # ROC and PR curves
    plot_roc_curves(curve_data)
    plot_pr_curves(curve_data)

    # Training loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Canonical quantum (bounded)", alpha=0.7)
    plt.plot(train_losses_no_ent, label="No entanglement", alpha=0.7)
    plt.plot(train_losses_unb, label="Unbounded embedding", alpha=0.7)
    plt.plot(train_losses_cl, label="Classical baseline", alpha=0.7)
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(config.PLOTS_DIR, "training_losses.png"), dpi=300)
    plt.close()
    print("  Training loss plot saved")

    # ── Comparison table ─────────────────────────────────────────────────
    comparison = []
    for exp_name, exp_results in all_results.items():
        comparison.append({
            "Experiment": exp_results["model_name"],
            "ROC-AUC": exp_results["roc_auc"],
            "PR-AUC": exp_results["pr_auc"],
            "Sensitivity": exp_results["sensitivity_recall"],
            "Specificity": exp_results["specificity"],
            "F1": exp_results["f1_score"],
            "MCC": exp_results["mcc"],
            "TP": exp_results["confusion_matrix"]["tp"],
            "FP": exp_results["confusion_matrix"]["fp"],
            "FN": exp_results["confusion_matrix"]["fn"],
            "TN": exp_results["confusion_matrix"]["tn"],
        })

    comparison_df = pd.DataFrame(comparison)
    comparison_csv = os.path.join(config.RESULTS_DIR, "experiment_comparison.csv")
    comparison_df.to_csv(comparison_csv, index=False)
    print(f"\n  Comparison table saved: {comparison_csv}")
    print("\n" + comparison_df.to_string(index=False))

    # ── Save all results JSON ────────────────────────────────────────────
    all_results["dataset_stats"] = dataset_stats
    all_results["versions"] = versions
    all_results["timestamp"] = datetime.datetime.now().isoformat()
    all_results["circuit_description"] = {
        "n_qubits": config.N_QUBITS,
        "embedding": "RX (explicit, bounded ∈ [0, π/2])",
        "variational": "RX (1 layer, 4 trainable params)",
        "entanglement": "CNOT ring: 0→1, 1→2, 2→3, 3→0",
        "measurement": "⟨PauliZ⟩ on all 4 qubits",
        "total_params": 109,
    }

    results_json = os.path.join(config.RESULTS_DIR, "all_results.json")
    with open(results_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Full results saved: {results_json}")

    # ── IonQ summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("IONQ EXPERIMENT SUMMARY")
    print("=" * 70)

    ionq_summary = f"""
IonQ Research Credit Application — Experiment Summary
======================================================

Project: Reproducible Quantum Simulation for CRISPR-Cas9 Off-Target Prediction
Architecture: 4-Qubit Hybrid Variational Quantum Classifier (109 parameters)

Dataset: Listgarten Dataset II/6 (22 gRNA, {dataset_stats['total_rows']} rows,
         {dataset_stats['total_positive']} positives, imbalance {dataset_stats['class_imbalance_ratio']})

Circuit:
  • 4 qubits
  • Classical bottleneck: Linear(24→4) + sigmoid × π/2
  • Data embedding: 4× RX gates (bounded angles ∈ [0, π/2])
  • Variational layer: 4× RX gates (trainable)
  • Entanglement: CNOT ring (0→1, 1→2, 2→3, 3→0)
  • Measurement: ⟨Z⟩ on all 4 qubits
  • Classical output: Linear(4→1)
  • Total trainable parameters: 109 (100 classical + 4 quantum + 5 classical)

Training: Adam (lr=0.005), BCEWithLogitsLoss(pos_weight=6816.11), 500 batches × 32

Results (full {dataset_stats['test_rows']}-sample test set):
  Ideal statevector:    ROC-AUC = {results_ideal['roc_auc']:.4f}, PR-AUC = {results_ideal['pr_auc']:.4f}
  Shot-based (1000):    ROC-AUC = {results_shots['roc_auc']:.4f}, PR-AUC = {results_shots['pr_auc']:.4f}
  No entanglement:      ROC-AUC = {results_no_ent['roc_auc']:.4f}, PR-AUC = {results_no_ent['pr_auc']:.4f}
  Unbounded embedding:  ROC-AUC = {results_unbounded['roc_auc']:.4f}, PR-AUC = {results_unbounded['pr_auc']:.4f}
  Classical baseline:   ROC-AUC = {results_classical['roc_auc']:.4f}, PR-AUC = {results_classical['pr_auc']:.4f}

Purpose of IonQ Hardware Execution:
  We seek to validate whether the trained 109-parameter hybrid quantum circuit
  produces reproducible classification behavior on real trapped-ion hardware.
  Specifically, we will compare:
    1. Ideal statevector simulation (PennyLane lightning.qubit)
    2. IonQ ideal simulator (cloud)
    3. IonQ noise model simulator
    4. IonQ Aria/Forte hardware
  on the same frozen weights and test sequences.

Software: Python {versions['python'].split()[0]}, PennyLane {versions['pennylane']},
          PyTorch {versions['torch']}, NumPy {versions['numpy']}
"""

    ionq_path = os.path.join(config.RESULTS_DIR, "ionq_summary.txt")
    with open(ionq_path, "w", encoding="utf-8") as f:
        f.write(ionq_summary)
    print(ionq_summary)
    print(f"  IonQ summary saved: {ionq_path}")

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nOutputs:")
    print(f"  Results:     {config.RESULTS_DIR}/")
    print(f"  Plots:       {config.PLOTS_DIR}/")
    print(f"  Checkpoints: {config.CHECKPOINTS_DIR}/")


if __name__ == "__main__":
    main()
