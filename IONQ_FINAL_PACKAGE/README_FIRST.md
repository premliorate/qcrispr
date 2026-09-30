# Reproducible Hybrid Quantum-Classical Simulation
## CRISPR Off-Target Prediction — IonQ Research Credit Application

### Problem
CRISPR-Cas9 guide RNA (gRNA) off-target site classification using the Listgarten Dataset II/6
(383,463 total samples, 56 positives, 22 gRNAs, 23 bp sequences, 24 mismatch features).

### Model
A frozen 4-qubit hybrid quantum-classical neural network with **109 total parameters**:

| Component | Details |
|---|---|
| Input | 24 mismatch features |
| Classical bottleneck | Linear(24→4) + sigmoid×(π/2) — 100 params |
| Quantum embedding | 4 × RX(xᵢ) — data encoding |
| Quantum variational | 4 × RX(θᵢ) — 4 trainable params |
| Entanglement | CNOT ring: 0→1→2→3→0 |
| Measurement | Pauli-Z on all 4 qubits |
| Output layer | Linear(4→1) — 5 params |
| Total | **109 parameters** |

### Verified Results

| Experiment | Samples | Shots | ROC-AUC | Notes |
|---|---|---|---|---|
| Local Ideal (statevector) | 76,693 | ∞ | 0.8704 | Full test set reference |
| Local Ideal (statevector) | 111 | ∞ | 0.8591 | IonQ comparison subset |
| IonQ Ideal Simulator | 111 | 1,000 | 0.8555 | Cloud-executed via API |
| IonQ Aria-1 Noise Model | 111 | 1,000 | 0.8818 | Cloud-executed via API |
| Classical Baseline | 76,693 | N/A | 0.9483 | 105-param parameter-matched |

> **Important:** The IonQ and classical results use different sample sizes and are **not** directly comparable.
> This study does **not** claim quantum advantage.

### Figures

| Figure | Description |
|---|---|
| `figures/roc_comparison.png` | ROC-AUC across all evaluation conditions |
| `figures/ionq_vs_local_scatter.png` | Local ideal vs IonQ simulator predictions (Pearson r = 0.9253) |
| `figures/canonical_circuit.png` | Canonical 4-qubit circuit diagram |
| `figures/shot_stability.png` | ROC-AUC and Pearson correlation vs shot count |
| `figures/model_architecture.png` | Full hybrid model architecture workflow |

### Limitations
- The full test set contains only **11 positive examples** out of 76,693, making full-test AUC sensitive to ranking of a very small number of positives.
- The 111-sample IonQ comparison set preserves all 11 positives for meaningful AUC comparison.
- IonQ cloud simulation results are distinct from IonQ QPU hardware results.

### Next Step
Upon allocation of research credits: execute the exact frozen circuit on IonQ Aria or Forte QPU hardware
and quantify hardware fidelity (MAE, Pearson r) relative to the validated simulation references.

### Files
See `IONQ_RESEARCH_SUMMARY.md` for the complete metrics table.
See `IONQ_RESEARCHER_DRAFT.md` for the researcher communication template.
