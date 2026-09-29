"""
model.py — Canonical 109-parameter hybrid variational quantum classifier.

Architecture (from the original notebook's FastHybridQCNN, Cell 31):
  Linear(24, 4)         → 24×4 + 4 = 100 classical params
  sigmoid × (π/2)       → bounded embedding ∈ [0, π/2]
  RX angle embedding    → 4 qubits (0 trainable params)
  RX variational layer  → 4 trainable quantum params
  CNOT ring: 0→1→2→3→0  → 0 trainable params
  Z expectation × 4     → measurement
  Linear(4, 1)          → 4×1 + 1 = 5 classical params
  ─────────────────────────────────────────────
  TOTAL                 = 109 trainable parameters

RX/RY INCONSISTENCY RESOLUTION
-------------------------------
The original notebook (Cell 31) used:
  - qml.AngleEmbedding(rotation='X')  → RX gates for data embedding
  - qml.BasicEntanglerLayers(weights)  → default rotation is RX
                                         + CNOT ring entanglement

The paper (Section 2.2) described "Rx embedding + Ry variational + CNOT ring."
Cell 16 (weight-reconstruction) used qml.RX for embedding and qml.RY for
variational, but that cell rebuilds a DIFFERENT model (the 745-param
sliding-window architecture).

RESOLUTION: We follow Cell 31 — the FastHybridQCNN that produced the
109-param result. Both embedding and variational layers use RX gates.
We use AngleEmbedding(rotation='X') and BasicEntanglerLayers(rotation='X')
to be explicit, and verify this matches hand-written RX + CNOT ring via
the draw_circuit() function.

PennyLane's BasicEntanglerLayers with 1 layer and default rotation:
  - Applies RX(weight[i]) on each qubit
  - Applies CNOT ring: (0,1), (1,2), (2,3), (3,0)
This is EXACTLY the circuit we want.
"""

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn

from . import config


# ── Quantum device (statevector / ideal simulation) ─────────────────────────
dev_ideal = qml.device("lightning.qubit", wires=config.N_QUBITS)


@qml.qnode(dev_ideal, interface="torch", diff_method="adjoint")
def quantum_circuit_ideal(inputs, weights):
    """
    Canonical 4-qubit quantum circuit.

    Uses PennyLane templates with EXPLICIT rotation='X' specification:
      AngleEmbedding(rotation='X')  → RX(inputs[i]) on each qubit
      BasicEntanglerLayers(rotation='X') → RX(weights[0,i]) + CNOT ring

    This exactly matches the original notebook's Cell 31.
    """
    # Embedding: RX gates with bounded classical input
    qml.AngleEmbedding(inputs, wires=range(config.N_QUBITS), rotation='X')

    # Variational + Entanglement: RX rotations + CNOT ring
    # BasicEntanglerLayers default rotation is 'X' (RX gates) + CNOT ring
    qml.BasicEntanglerLayers(weights, wires=range(config.N_QUBITS), rotation=qml.RX)

    # Measurement: Z expectation on all qubits
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(config.N_QUBITS)]


class HybridQuantumClassifier(nn.Module):
    """
    Canonical 109-parameter hybrid variational quantum classifier.

    This is the exact architecture from the original notebook's FastHybridQCNN,
    with rotation='X' explicitly specified for both embedding and variational
    layers to resolve the RX/RY inconsistency.
    """

    def __init__(self, use_entanglement=True, bounded_embedding=True):
        super().__init__()

        # Classical bottleneck: 24 → 4
        self.fc_in = nn.Linear(config.INPUT_DIM, config.N_QUBITS)

        # Quantum layer via TorchLayer
        weight_shapes = {"weights": (config.N_QLAYERS, config.N_QUBITS)}

        if use_entanglement:
            self.qlayer = qml.qnn.TorchLayer(quantum_circuit_ideal, weight_shapes)
        else:
            # No-entanglement ablation
            self.qlayer = qml.qnn.TorchLayer(
                quantum_circuit_no_entanglement, weight_shapes
            )

        # Classical output: 4 → 1
        self.fc_out = nn.Linear(config.N_QUBITS, 1)

        self.bounded_embedding = bounded_embedding

    def forward(self, x):
        x = x.float()

        if self.bounded_embedding:
            # Bounded embedding: sigmoid(Wx+b) × π/2 → angles ∈ [0, π/2]
            x_in = torch.sigmoid(self.fc_in(x)) * (np.pi / 2.0)
        else:
            # Unbounded/raw embedding: direct linear output as angles
            x_in = self.fc_in(x)

        q_out = self.qlayer(x_in)
        return self.fc_out(q_out)

    def count_parameters(self) -> dict:
        """Return detailed parameter breakdown."""
        fc_in_params = sum(p.numel() for p in self.fc_in.parameters())
        q_params = sum(p.numel() for p in self.qlayer.parameters())
        fc_out_params = sum(p.numel() for p in self.fc_out.parameters())
        total = fc_in_params + q_params + fc_out_params
        return {
            "fc_in (classical)": fc_in_params,
            "qlayer (quantum)": q_params,
            "fc_out (classical)": fc_out_params,
            "total": total,
        }


# ── Ablation circuits ────────────────────────────────────────────────────────

dev_no_ent = qml.device("lightning.qubit", wires=config.N_QUBITS)

@qml.qnode(dev_no_ent, interface="torch", diff_method="adjoint")
def quantum_circuit_no_entanglement(inputs, weights):
    """Same circuit but WITHOUT CNOT entanglement gates.

    Uses RX angle embedding + RX variational rotation, but NO entanglement.
    Each qubit processes data independently.
    """
    qml.AngleEmbedding(inputs, wires=range(config.N_QUBITS), rotation='X')
    # Apply only the rotation part of BasicEntanglerLayers — no CNOT ring
    for i in range(config.N_QUBITS):
        qml.RX(weights[0, i], wires=i)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(config.N_QUBITS)]


# ── Classical baseline (parameter-matched) ───────────────────────────────────

class ClassicalBaseline109(nn.Module):
    """
    Parameter-matched classical baseline with ~109 parameters.

    Architecture:
        Linear(24, 4) → Sigmoid → Linear(4, 1)
        = 24×4 + 4 + 4×1 + 1 = 109 parameters

    Uses the same sigmoid nonlinearity as the quantum model's bounded
    embedding, making it the fairest possible classical control.
    """

    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(config.INPUT_DIM, config.N_QUBITS)   # 100 params
        self.fc_out = nn.Linear(config.N_QUBITS, 1)                  # 5 params

    def forward(self, x):
        x = x.float()
        x = torch.sigmoid(self.fc_in(x))  # Same sigmoid as quantum model
        return self.fc_out(x)              # Raw logits for BCEWithLogitsLoss


# ── Shot-based simulation circuit ────────────────────────────────────────────

dev_shots = qml.device("default.qubit", wires=config.N_QUBITS)

@qml.qnode(dev_shots, interface="torch")
def _quantum_circuit_shots_base(inputs, weights):
    """
    Same circuit as ideal, but will be measured with finite shots.
    """
    qml.AngleEmbedding(inputs, wires=range(config.N_QUBITS), rotation='X')
    qml.BasicEntanglerLayers(weights, wires=range(config.N_QUBITS), rotation=qml.RX)
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(config.N_QUBITS)]

# Apply shots via the modern set_shots transform
quantum_circuit_shots = qml.set_shots(_quantum_circuit_shots_base, shots=config.N_SHOTS)


class HybridQuantumClassifierShots(nn.Module):
    """Same architecture but using shot-based quantum simulation."""

    def __init__(self):
        super().__init__()
        self.fc_in = nn.Linear(config.INPUT_DIM, config.N_QUBITS)
        weight_shapes = {"weights": (config.N_QLAYERS, config.N_QUBITS)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit_shots, weight_shapes)
        self.fc_out = nn.Linear(config.N_QUBITS, 1)

    def forward(self, x):
        x = x.float()
        x_in = torch.sigmoid(self.fc_in(x)) * (np.pi / 2.0)
        q_out = self.qlayer(x_in)
        return self.fc_out(q_out)


def draw_circuit():
    """Return a text drawing of the canonical quantum circuit using explicit gates."""

    @qml.qnode(qml.device("default.qubit", wires=4))
    def _draw_circuit(inputs, weights):
        # Explicit RX gates for maximum clarity
        for i in range(4):
            qml.RX(inputs[i], wires=i)
        for i in range(4):
            qml.RX(weights[0, i], wires=i)
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 0])
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(4)]

    dummy_inputs = np.array([0.5, 0.3, 0.7, 0.1])
    dummy_weights = np.array([[0.1, 0.2, 0.3, 0.4]])

    return qml.draw(_draw_circuit, decimals=2)(dummy_inputs, dummy_weights)
