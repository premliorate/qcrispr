import json
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import HybridQuantumClassifier

def main():
    model = HybridQuantumClassifier()
    counts = model.count_parameters()
    
    # Verification
    expected = {
        "fc_in": 100,
        "qlayer": 4,
        "fc_out": 5,
        "total": 109
    }
    
    actual = {
        "fc_in": counts["fc_in (classical)"],
        "qlayer": counts["qlayer (quantum)"],
        "fc_out": counts["fc_out (classical)"],
        "total": counts["total"]
    }
    
    match = (actual == expected)
    
    result = {
        "expected": expected,
        "actual": actual,
        "verified": match
    }
    
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"), exist_ok=True)
    with open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "parameter_verification.json"), "w") as f:
        json.dump(result, f, indent=2)
        
    if not match:
        print("FAIL: Parameter counts do not match!")
        sys.exit(1)
    else:
        print("PASS: Parameter counts verified.")

if __name__ == "__main__":
    main()
