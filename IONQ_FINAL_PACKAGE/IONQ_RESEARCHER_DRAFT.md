Subject: Research Credit Application — Hybrid Quantum Classifier Hardware Fidelity Study
         (CRISPR Off-Target Prediction, 4 Qubits, 109 Parameters, Frozen Weights)

Dear IonQ Research Team,

We are seeking IonQ research credits to execute a frozen hybrid quantum-classical network
on IonQ hardware and quantify hardware fidelity relative to our validated simulation baseline.

---

**Project:** CRISPR-Cas9 Off-Target Prediction
**Dataset:** Listgarten Dataset II/6 (383,463 samples, 56 positives, 22 gRNAs)

**Architecture (109 total parameters, completely frozen):**
- 24-dimensional mismatch encoding → Linear(24→4) classical bottleneck (100 params)
- sigmoid × (π/2) normalization to bound inputs to [0, π/2]
- 4 × RX(xᵢ) data embedding gates
- 4 × RX(θᵢ) trainable variational gates (4 params)
- 4-CNOT ring entanglement structure (q0→q1→q2→q3→q0)
- 4 Pauli-Z expectation values as quantum output
- Linear(4→1) classical output layer (5 params)

**Verified IonQ Cloud Simulation Results (111-sample stratified comparison set):**

| Condition | ROC-AUC | Pearson r vs local ideal |
|---|---|---|
| Local ideal (statevector) | 0.8591 | 1.000 |
| IonQ ideal simulator | 0.8555 | 0.9253 |
| IonQ Aria-1 noise model | 0.8818 | 0.7563 |

Local full-test reference (76,693 samples, statevector): **ROC-AUC = 0.8704**

The classical parameter-matched baseline achieves ROC-AUC 0.9483.
**We do not claim quantum advantage.** Our objective is to characterise the hardware-fidelity
gap between ideal statevector simulation and trapped-ion QPU execution on a real bioinformatics task.

**Request:**
We are seeking IonQ research credits to execute the same frozen circuit and parameters on
IonQ hardware and quantify hardware fidelity relative to the validated simulation reference.
The circuit has been formally verified (equivalence PASS, parameter count PASS) and all
weights are frozen. No retraining will be performed on hardware.

We look forward to discussing this further.

Sincerely,
[Researcher Name]
[Affiliation]
