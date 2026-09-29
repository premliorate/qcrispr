# IonQ Research Simulation Report
**Project:** Reproducible Quantum Simulation for CRISPR-Cas9 Off-Target Prediction

## 1. Project Objective
This project prepares a reproducible, frozen hybrid quantum-classical neural network for execution on IonQ hardware. The objective is to validate whether the exact 109-parameter model produces consistent classification behavior across local ideal simulators, IonQ's ideal simulator, IonQ's noise-model simulators, and eventual IonQ trapped-ion QPUs.

## 2. Exact Dataset
- **Source:** Listgarten Dataset II/6
- **Total rows:** 383,463
- **Positives (off-targets):** 56
- **Negatives (safe):** 383,407
- **Class imbalance:** 1:6,846
- **Sequence properties:** 22 unique gRNAs, 23 bp sequence length.

## 3. Exact Preprocessing
- Binary mismatch encoding: `match → 0.0`, `mismatch → π`.
- Sequences are zero-padded to 24 input features.

## 4. Exact Split
- **Method:** `train_test_split` from `sklearn` (Stratified by Label).
- **Test Size:** 20% (76,693 samples).
- **Seed:** `random_state=42`.
- **Test set composition:** 11 positives, 76,682 negatives.

## 5. Frozen Checkpoint
- **File:** `checkpoints/canonical_109param_weights.pth`
- The model weights are completely frozen. There is no retraining on the IonQ simulators or QPUs.

## 6. Complete 109-Parameter Architecture
- **fc_in:** `Linear(24 → 4)` (100 parameters)
- **Bounding:** Sigmoid activation scaled by $\pi/2$
- **qlayer:** 4-qubit quantum circuit (4 parameters)
- **fc_out:** `Linear(4 → 1)` (5 parameters)
- **Total:** 109 trainable parameters.

## 7. Gate-by-Gate Circuit
```
0: ──RX(data_0)──RX(theta_0)─╭●───────╭X─┤  ⟨Z⟩
1: ──RX(data_1)──RX(theta_1)─╰X─╭●────│──┤  ⟨Z⟩
2: ──RX(data_2)──RX(theta_2)────╰X─╭●─│──┤  ⟨Z⟩
3: ──RX(data_3)──RX(theta_3)───────╰X─╰●─┤  ⟨Z⟩
```
*Where `data_i` are bounded classical inputs $\in [0, \pi/2]$ and `theta_i` are the 4 trainable parameters.*

## 8. Why this circuit was selected
This circuit was selected because its 4-qubit requirement perfectly matches the classical `24→4` dimensionality reduction bottleneck. The single RX layer keeps the trainable parameter count extremely low (4 quantum parameters), while the CNOT ring ensures full inter-qubit correlation prior to the $\langle Z \rangle$ measurement. The architecture is compact enough to run reliably on near-term hardware.

## 9. Local Ideal Results
Evaluated on the full 76,693-sample test set using PennyLane's `lightning.qubit`:
- **ROC-AUC:** 0.8704
- **PR-AUC:** 0.0117

## 10. Shot-Based Results (1000 shots)
Evaluated on the full test set using PennyLane's `default.qubit` with 1000 shots:
- **ROC-AUC:** 0.9263
- **PR-AUC:** 0.0153
- **Pearson correlation vs Ideal:** 0.8435

## 11. IonQ Simulator Results
*(Evaluated via `ionq.simulator` via Pennylane-IonQ. See `results/ionq_sim_results.json` for live execution data)*
- Using `ionq.simulator`, we ensure that the API transmission, circuit transpilation, and remote execution logic are sound.

## 12. IonQ Noise-Model Results
- By targeting `noise_model="aria-1"` on the `ionq.simulator`, we approximate the depolarizing and thermal relaxation errors of the Aria-1 QPU.

## 13. Classical Baseline
A parameter-matched classical model (`Linear(24→4) → Sigmoid → Linear(4→1)`, 105 total parameters) was evaluated for rigorous benchmarking.
- **ROC-AUC:** 0.9483
- **PR-AUC:** 0.0364

## 14. Ablation Results
- **No Entanglement:** ROC-AUC = 0.9171
- **Unbounded Embedding:** ROC-AUC = 0.9212

## 15. Statistical Limitation
The test set contains only **11 positive examples** against 76,682 negatives. This extreme imbalance is an inherent property of the Listgarten Dataset II/6. Small fluctuations in the ranking of those 11 positives can cause significant swings in the ROC-AUC. 

## 16. Quantum Advantage Disclaimer
**This canonical experiment does not demonstrate quantum advantage.** The 105-parameter classical baseline strictly outperforms the 109-parameter hybrid quantum model. Furthermore, removing the entanglement CNOT ring improves the quantum model's ROC-AUC. The value of this experiment lies in studying the hardware fidelity of a frozen, classically-outperformed quantum model.

## 17. Next Experiment
The exact next step upon receipt of IonQ research credits is the **frozen-circuit execution on IonQ Aria/Forte hardware**. We will run the exact circuit, with the exact frozen weights, across the test set (or a representative subsample) to measure QPU deviations from the ideal statevector simulation outputs.
