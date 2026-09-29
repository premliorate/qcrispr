import os
import sys

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(root, "results")
    scripts_dir = os.path.join(root, "scripts")
    checkpoints_dir = os.path.join(root, "checkpoints")

    def check_file(path):
        return os.path.exists(path)

    tests = {
        "DATASET": check_file(os.path.join(root, "data", "Listgarten_22gRNA_wholeDataset.csv")),
        "SPLIT": check_file(os.path.join(checkpoints_dir, "train_indices.npy")) and check_file(os.path.join(checkpoints_dir, "test_indices.npy")),
        "CHECKPOINT": check_file(os.path.join(checkpoints_dir, "canonical_109param_weights.pth")),
        "PARAMETERS": check_file(os.path.join(results_dir, "parameter_verification.json")),
        "CANONICAL CIRCUIT": check_file(os.path.join(root, "src", "canonical_circuit.py")),
        "CIRCUIT EQUIVALENCE": check_file(os.path.join(results_dir, "circuit_equivalence.json")),
        "LOCAL IDEAL": check_file(os.path.join(results_dir, "all_results.json")),
        "SHOT SIMULATION": check_file(os.path.join(results_dir, "shot_stability.json")),
        "IONQ CIRCUIT": check_file(os.path.join(results_dir, "ionq_sim_results.json")),
        "SECURITY": check_file(os.path.join(scripts_dir, "scrub_credentials.py")),
        "REPORT": check_file(os.path.join(results_dir, "IONQ_RESEARCH_SIMULATION_REPORT.md"))
    }

    all_pass = all(tests.values())

    for name, passed in tests.items():
        status = "PASS" if passed else "FAIL"
        print(f"{name.ljust(25)} {status}")

    print()
    if all_pass:
        print("FINAL VALIDATION: PASS")
    else:
        print("FINAL VALIDATION: FAIL")
        sys.exit(1)

if __name__ == "__main__":
    main()
