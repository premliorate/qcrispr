# Reproducible Hybrid Quantum-Classical Simulation for CRISPR Off-Target Prediction

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
- **Local Reference Results (111-Sample Comparison Set)**: ROC-AUC = 0.8591
- **IonQ Ideal Simulation Results (111-Sample Comparison Set)**: ROC-AUC = 0.8555 (Correlation vs local: 0.9253)
- **IonQ Noise-Model Results (Aria-1, 111-Sample Comparison Set)**: ROC-AUC = 0.8818

## Limitations
The full held-out test set contains only 11 positive examples out of 76,693 total examples. Because of this extreme class imbalance, full-test AUC estimates are highly sensitive to ranking changes among a very small number of positives. Furthermore, this study does not demonstrate quantum advantage; the parameter-matched classical baseline outperforms the quantum model.

## Next Steps
Upon allocation of research credits, the next step is QPU execution on IonQ hardware to quantify hardware fidelity relative to the validated simulation references established in this package.
