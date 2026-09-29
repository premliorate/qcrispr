# IonQ Research Summary
    
| Experiment | Backend | Samples | Shots | ROC-AUC | PR-AUC | Correlation vs Local |
| --- | --- | --- | --- | --- | --- | --- |
| Local Ideal Full | lightning.qubit | 76693 | N/A (Statevector) | 0.8704 | 0.0117 | N/A |
| Local Ideal Subsample | lightning.qubit | 111 | N/A (Statevector) | 0.8591 | N/A | 1.0000 |
| Classical Baseline | CPU (PyTorch) | 76693 | N/A | 0.9483 | 0.0364 | N/A |
| IonQ Ideal | ionq.simulator | 111 | 1000 | 0.8555 | 0.6356 | 0.9253 |
| IonQ Noise (Aria-1) | ionq.simulator | 111 | 1000 | 0.8818 | 0.5225 | 0.7563 |

*Explicitly noted: this study does not demonstrate quantum advantage. The parameter-matched classical baseline (ROC-AUC 0.9483) outperforms the quantum model. The goal is to study hardware inference fidelity of a frozen network.*

## Circuit Claims
- **4 qubits**: Selected to match the 24->4 classical dimensionality reduction bottleneck.
- **Shallow depth**: A single variational layer was chosen to keep the model compact for hardware fidelity study.
- **RX embedding**: Direct encoding of the 4 latent features.
- **CNOT ring**: Explicit entangling hypothesis tested by ablation. (Ablation showed removing the CNOT ring produced higher ROC-AUC in ideal simulation, indicating the structure did not improve predictive performance for this particular model configuration).
- **Pauli-Z measurements**: Provide a 4-dimensional quantum feature vector to the final classical layer.
