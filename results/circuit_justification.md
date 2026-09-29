# Hybrid Quantum Circuit Justification

This document provides the scientific justification for the chosen 4-qubit Hybrid Variational Quantum Classifier (109 total parameters). It focuses strictly on the evidence supported by the simulation results and the architectural constraints of the model.

## 1. The Classical-Quantum Interface

The architecture utilizes a classical bottleneck `Linear(24 → 4)` to reduce the high-dimensional zero-padded CRISPR mismatch sequence (24 features) into a 4-dimensional latent space. This choice directly dictates the requirement for **4 qubits**. A sigmoid activation scaled by $\pi/2$ ensures that the outputs of this bottleneck are bounded $\in [0, \pi/2]$, matching the valid domain for rotation gates (preventing phase wrap-around).

## 2. Quantum Data Embedding

We employ an **RX Angle Embedding**, where each of the 4 classical latent features is mapped to the X-rotation of a corresponding qubit:
$$ RX(\theta_i) $$
This provides a non-linear mapping of the classical data into the Hilbert space.

## 3. Trainable Variational Layer

A single trainable layer of **RX rotations** (4 parameters total) follows the embedding layer.
- **Why a single layer?** A shallow circuit minimizes the number of trainable parameters (only 4 in the quantum circuit). As a design choice, this targets mitigating trainability issues often seen in deep variational circuits.
- **Why RX?** This matches the data embedding axis, allowing the model to apply a trainable shift to the feature rotations before entanglement.

## 4. Entanglement Strategy

We use a **CNOT ring** structure ($0 \rightarrow 1, 1 \rightarrow 2, 2 \rightarrow 3, 3 \rightarrow 0$) to introduce inter-qubit interactions. This ring architecture is hardware-efficient on many platforms and ensures that every qubit's state is conditionally dependent on its neighbor, spreading the classical feature information across the full 16-dimensional basis of the 4-qubit system.

## 5. Measurement and Output

The circuit measures the **Pauli-Z expectation $\langle Z \rangle$** on all 4 qubits. These 4 expectation values form the quantum feature vector, which is passed to the final classical layer `Linear(4 → 1)` (5 parameters) to produce the raw logit for binary classification.

## Total Parameter Count
- `fc_in`: 100 classical parameters
- `quantum layer`: 4 quantum parameters
- `fc_out`: 5 classical parameters
- **Total: 109 trainable parameters**

## Scientific Constraints and Claims

1. **No Claims of Quantum Advantage:** The 109-parameter quantum model (ROC-AUC ~0.87) is currently outperformed by the equivalent parameter-matched classical baseline (105-parameter `Linear(24→4) → Sigmoid → Linear(4→1)`, ROC-AUC ~0.94). The goal of running this on IonQ hardware is **not** to demonstrate superiority, but to validate hardware fidelity and reproducibility of this specific hybrid structure.
2. **Entanglement Role:** The ablation showed that removing the CNOT ring produced higher ROC-AUC in the present ideal-simulation experiment. This indicates that the selected entangling structure did not improve predictive performance for this particular model configuration.
3. **Hardware Execution Goal:** The justification for this circuit is its compactness and trainability. The next step is to freeze these weights and run the exact circuit on IonQ hardware to measure finite-sampling and hardware-noise deviations from the ideal statevector simulation.
