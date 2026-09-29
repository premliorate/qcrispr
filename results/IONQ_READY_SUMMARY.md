# IONQ PACKAGE READY

## 1. What exact circuit are we giving IonQ?
We are submitting a 4-qubit Hybrid Variational Quantum Classifier with 109 trainable parameters. 
The explicit circuit operates on 4 qubits and consists of:
- **Angle Embedding**: 4 parallel RX rotations, mapping the classical bottleneck output (bounded in $[0, \pi/2]$) into the quantum state.
- **Variational Layer**: 4 parallel trainable RX rotations.
- **Entanglement Ring**: A sequence of 4 CNOT gates ($0 \rightarrow 1, 1 \rightarrow 2, 2 \rightarrow 3, 3 \rightarrow 0$).
- **Measurement**: Pauli-Z expectation values on all 4 qubits.

## 2. Why was this circuit selected?
This circuit was selected for its extreme parameter efficiency and shallow depth. The 4-qubit configuration precisely matches the classical $24 \rightarrow 4$ dimensionality reduction bottleneck, while the single layer of trainable RX rotations keeps the quantum parameter count to a minimum (4 parameters). The CNOT ring distributes correlations across the 4-qubit Hilbert space. Because the classically matched architecture actually outperforms the quantum one at this scale (ROC-AUC 0.94 vs 0.87), our focus is not on claiming quantum advantage, but strictly on utilizing this compact, trainable circuit to validate and benchmark the hardware fidelity of trapped-ion QPUs.

## 3. What exact dataset, preprocessing, split, and frozen weights were used?
- **Dataset**: The complete Listgarten Dataset II/6 (383,463 rows, 56 positive mismatch off-targets, 22 unique gRNAs).
- **Preprocessing**: Binary mismatch encoding (match = 0.0, mismatch = $\pi$), zero-padded to 24 input features.
- **Split**: 80/20 Stratified `train_test_split` with `random_state=42`. The test set consists of 76,693 rows containing 11 positive examples.
- **Frozen Weights**: The `canonical_109param_weights.pth` file containing exactly 109 parameters. These weights are completely frozen and will not be fine-tuned.

## 4. What simulation results were obtained?
Across the 76,693 test samples, using the frozen checkpoint:
- **Local Ideal Statevector**: ROC-AUC = 0.8704, PR-AUC = 0.0117 (full test set)
- **Shot-Based (1,000 shots, full set)**: ROC-AUC = 0.9263, PR-AUC = 0.0153 (all_results.json reference)
- **Classical Baseline (105 params)**: ROC-AUC = 0.9483, PR-AUC = 0.0364
- **IonQ Ideal Simulator (Subsample 111 samples)**: ROC-AUC = 0.8555 (vs 0.8591 local ideal). Pearson correlation vs local ideal: 0.9253.
- **IonQ Noise Simulator (Aria-1)**: ROC-AUC = 0.8818. Successfully simulated Aria-1 depolarizing/thermal relaxation noise.

## 5. What exactly will be run on IonQ hardware after credits are granted?
After receiving research credits, we will deploy the exact canonical circuit described above with the exact frozen weights to the **IonQ Aria or Forte QPUs**. We will run the deterministic test inputs (or a representative stratified subsample) to evaluate hardware inference performance. The objective is to measure the deviation (via metrics such as MAE, Pearson correlation) between the physical hardware outputs and the idealized statevector outputs, providing a direct assessment of near-term quantum hardware fidelity on a highly imbalanced bioinformatics task.

---

### Package Output Summary

- **Local Ideal (Full)**: ROC-AUC = 0.8704
- **IonQ Ideal (Subsample)**: ROC-AUC = 0.8555 (Correlation 0.925)
- **Total Parameters**: 109
- **IonQ Simulator Status**: PASS (API Integrated & Tested)
- **Noise Model Status**: PASS (Aria-1 Target Executed)
- **Final Validation Status**: PASS

### Final Deliverable Files
- `results/IONQ_RESEARCH_SIMULATION_REPORT.md` (Comprehensive scientific report)
- `results/circuit_justification.md` (Circuit analysis)
- `results/circuit_equivalence.json` (Circuit verification trace)
- `results/parameter_verification.json` (Parameter count trace)
- `results/reproducibility_manifest.json` (SHA-256 hashes and dependencies)
- `results/shot_stability.json` (Shot count analysis)
- `src/canonical_circuit.py` (Source of truth circuit definition)
- `checkpoints/canonical_109param_weights.pth` (Frozen weights)
