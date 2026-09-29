"""
canonical_circuit.py — The single authoritative source of truth for the 4-qubit Hybrid Variational Quantum Classifier circuit.

This file provides the explicit, gate-by-gate definition of the quantum circuit.
"""
import pennylane as qml

N_QUBITS = 4

def build_canonical_circuit(inputs, weights):
    """
    Explicit, gate-by-gate definition of the quantum circuit.
    
    Args:
        inputs: Array-like of length 4, containing bounded angles in [0, pi/2]
        weights: Array-like of shape (1, 4), containing trainable parameters
        
    Returns:
        List of expectation values of PauliZ on all 4 qubits.
    """
    # 1. Data embedding: RX gates
    for i in range(N_QUBITS):
        qml.RX(inputs[i], wires=i)
        
    # 2. Trainable variational layer: RX gates
    for i in range(N_QUBITS):
        qml.RX(weights[0, i], wires=i)
        
    # 3. Entanglement: CNOT ring
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 0])
    
    # 4. Measurement: PauliZ expectation on all qubits
    return [qml.expval(qml.PauliZ(wires=i)) for i in range(N_QUBITS)]

# Create a generic device and qnode for easy visualization/testing
dev = qml.device("default.qubit", wires=N_QUBITS)
canonical_qnode = qml.QNode(build_canonical_circuit, dev, interface="torch")
