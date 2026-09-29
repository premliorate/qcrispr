import json
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import quantum_circuit_ideal
from src.canonical_circuit import canonical_qnode

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Test cases: combinations of deterministic inputs and weights
    test_cases = [
        (torch.tensor([0.0, 0.0, 0.0, 0.0]), torch.tensor([[0.0, 0.0, 0.0, 0.0]])),
        (torch.tensor([np.pi/2, np.pi/2, np.pi/2, np.pi/2]), torch.tensor([[0.1, 0.2, 0.3, 0.4]])),
        (torch.tensor([0.1, 0.5, 1.0, 1.5]), torch.tensor([[-0.1, 0.5, 1.2, -1.0]])),
        (torch.rand(4) * np.pi/2, torch.rand(1, 4) * 2 * np.pi)
    ]
    
    results = []
    all_match = True
    
    for i, (inputs, weights) in enumerate(test_cases):
        # We need to evaluate the qnodes.
        # quantum_circuit_ideal takes (inputs, weights)
        res_model = quantum_circuit_ideal(inputs, weights)
        if isinstance(res_model, (list, tuple)):
            res_model = torch.stack(res_model)
            
        res_canonical = canonical_qnode(inputs, weights)
        if isinstance(res_canonical, (list, tuple)):
            res_canonical = torch.stack(res_canonical)
            
        # Check equivalence
        diff = torch.abs(res_model - res_canonical).max().item()
        match = diff < 1e-5
        
        results.append({
            "test_case": i + 1,
            "inputs": inputs.tolist(),
            "weights": weights.tolist(),
            "model_output": res_model.tolist(),
            "canonical_output": res_canonical.tolist(),
            "max_difference": diff,
            "match": match
        })
        if not match:
            all_match = False

    output = {
        "status": "PASS" if all_match else "FAIL",
        "details": results
    }
    
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "circuit_equivalence.json"), "w") as f:
        json.dump(output, f, indent=2)
        
    print(f"Circuit Equivalence: {output['status']}")
    if not all_match:
        sys.exit(1)

if __name__ == "__main__":
    main()
