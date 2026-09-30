import json, os, shutil, hashlib, re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = os.path.join(ROOT, "IONQ_FINAL_PACKAGE")
FIG_SRC = os.path.join(ROOT, "figures")

with open(os.path.join(ROOT, "results", "ionq_sim_results.json")) as f:
    ionq = json.load(f)
with open(os.path.join(ROOT, "results", "shot_stability.json")) as f:
    stab = json.load(f)
with open(os.path.join(ROOT, "results", "all_results.json")) as f:
    all_res = json.load(f)

LOCAL_IDEAL_ROC = stab["ideal"]["roc_auc"]
LOCAL_111_ROC   = ionq["local_ideal_probs_recomputed_roc"]
IONQ_IDEAL_ROC  = ionq["ionq_ideal_simulator"]["roc_auc"]
IONQ_NOISE_ROC  = ionq["ionq_noise_simulator"]["roc_auc"]
IONQ_IDEAL_PR   = ionq["ionq_ideal_simulator"]["pr_auc"]
IONQ_NOISE_PR   = ionq["ionq_noise_simulator"]["pr_auc"]
PEARSON         = ionq["ionq_ideal_simulator"]["pearson_corr_vs_ideal"]
NOISE_PEARSON   = ionq["ionq_noise_simulator"]["pearson_corr_vs_ideal"]
CLASSICAL_ROC   = all_res["experiment_5_classical_baseline"]["roc_auc"]
CLASSICAL_PR    = all_res["experiment_5_classical_baseline"]["pr_auc"]

def write(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")

def get_hash(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def check_no_credentials(folder):
    bad = [r"zXs7thhwvFvvimqgCLwzySXO3ADaZrrY", r"(?i)api_key\s*=\s*['\"][a-zA-Z0-9]{20,}['\"]"]
    for dname, _, fnames in os.walk(folder):
        for fname in fnames:
            if fname.endswith((".py", ".md", ".json", ".txt", ".csv")):
                fpath = os.path.join(dname, fname)
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                    for p in bad:
                        if re.search(p, content):
                            print(f"[FAIL] Credential found in {fpath}")
                            return False
    return True

# ── Rebuild package dir ──────────────────────────────────────────────────────
if os.path.exists(PKG_DIR):
    shutil.rmtree(PKG_DIR)
for d in [PKG_DIR, os.path.join(PKG_DIR,"src"), os.path.join(PKG_DIR,"scripts"),
          os.path.join(PKG_DIR,"checkpoints"), os.path.join(PKG_DIR,"figures")]:
    os.makedirs(d, exist_ok=True)

# ── README_FIRST.md ──────────────────────────────────────────────────────────
readme = f"""# Reproducible Hybrid Quantum-Classical Simulation
## CRISPR Off-Target Prediction — IonQ Research Credit Application

### Problem
CRISPR-Cas9 guide RNA (gRNA) off-target site classification using the Listgarten Dataset II/6
(383,463 total samples, 56 positives, 22 gRNAs, 23 bp sequences, 24 mismatch features).

### Model
A frozen 4-qubit hybrid quantum-classical neural network with **109 total parameters**:

| Component | Details |
|---|---|
| Input | 24 mismatch features |
| Classical bottleneck | Linear(24→4) + sigmoid×(π/2) — 100 params |
| Quantum embedding | 4 × RX(xᵢ) — data encoding |
| Quantum variational | 4 × RX(θᵢ) — 4 trainable params |
| Entanglement | CNOT ring: 0→1→2→3→0 |
| Measurement | Pauli-Z on all 4 qubits |
| Output layer | Linear(4→1) — 5 params |
| Total | **109 parameters** |

### Verified Results

| Experiment | Samples | Shots | ROC-AUC | Notes |
|---|---|---|---|---|
| Local Ideal (statevector) | 76,693 | ∞ | {LOCAL_IDEAL_ROC:.4f} | Full test set reference |
| Local Ideal (statevector) | 111 | ∞ | {LOCAL_111_ROC:.4f} | IonQ comparison subset |
| IonQ Ideal Simulator | 111 | 1,000 | {IONQ_IDEAL_ROC:.4f} | Cloud-executed via API |
| IonQ Aria-1 Noise Model | 111 | 1,000 | {IONQ_NOISE_ROC:.4f} | Cloud-executed via API |
| Classical Baseline | 76,693 | N/A | {CLASSICAL_ROC:.4f} | 105-param parameter-matched |

> **Important:** The IonQ and classical results use different sample sizes and are **not** directly comparable.
> This study does **not** claim quantum advantage.

### Figures

| Figure | Description |
|---|---|
| `figures/roc_comparison.png` | ROC-AUC across all evaluation conditions |
| `figures/ionq_vs_local_scatter.png` | Local ideal vs IonQ simulator predictions (Pearson r = {PEARSON:.4f}) |
| `figures/canonical_circuit.png` | Canonical 4-qubit circuit diagram |
| `figures/shot_stability.png` | ROC-AUC and Pearson correlation vs shot count |
| `figures/model_architecture.png` | Full hybrid model architecture workflow |

### Limitations
- The full test set contains only **11 positive examples** out of 76,693, making full-test AUC sensitive to ranking of a very small number of positives.
- The 111-sample IonQ comparison set preserves all 11 positives for meaningful AUC comparison.
- IonQ cloud simulation results are distinct from IonQ QPU hardware results.

### Next Step
Upon allocation of research credits: execute the exact frozen circuit on IonQ Aria or Forte QPU hardware
and quantify hardware fidelity (MAE, Pearson r) relative to the validated simulation references.

### Files
See `IONQ_RESEARCH_SUMMARY.md` for the complete metrics table.
See `IONQ_RESEARCHER_DRAFT.md` for the researcher communication template.
"""
write(os.path.join(PKG_DIR, "README_FIRST.md"), readme)

# ── IONQ_RESEARCH_SUMMARY.md ─────────────────────────────────────────────────
summary = f"""# IonQ Research Summary
## Frozen 4-Qubit Hybrid Quantum Classifier — CRISPR Off-Target Prediction

### Results Table

| Experiment | Backend | Samples | Shots | ROC-AUC | PR-AUC | Pearson vs Local |
|---|---|---|---|---|---|---|
| Local Ideal (statevector) | lightning.qubit | 76,693 | ∞ | {LOCAL_IDEAL_ROC:.4f} | {stab["ideal"]["pr_auc"]:.4f} | N/A |
| Local Ideal (statevector) | lightning.qubit | 111 | ∞ | {LOCAL_111_ROC:.4f} | N/A | 1.0000 |
| Classical Baseline | CPU (PyTorch) | 76,693 | N/A | {CLASSICAL_ROC:.4f} | {CLASSICAL_PR:.4f} | N/A |
| IonQ Ideal | ionq.simulator | 111 | 1,000 | {IONQ_IDEAL_ROC:.4f} | {IONQ_IDEAL_PR:.4f} | {PEARSON:.4f} |
| IonQ Noise (Aria-1) | ionq.simulator | 111 | 1,000 | {IONQ_NOISE_ROC:.4f} | {IONQ_NOISE_PR:.4f} | {NOISE_PEARSON:.4f} |

### Circuit Design Rationale

| Design Choice | Rationale |
|---|---|
| **4 qubits** | Matches the 24→4 classical dimensionality reduction bottleneck |
| **Shallow depth** | Design choice to keep circuit compact for hardware fidelity study |
| **RX embedding** | Direct encoding of the 4 latent features |
| **CNOT ring** | Explicit entangling hypothesis tested by ablation (ablation showed removing CNOT produced higher ROC-AUC in ideal simulation; the entangling structure did not improve predictive performance for this configuration) |
| **Pauli-Z measurements** | 4-dimensional quantum feature vector fed to classical output |

### Key Findings
- IonQ ideal cloud simulation achieved Pearson r = **{PEARSON:.4f}** vs local statevector ideal
- IonQ Aria-1 noise-model simulation achieved Pearson r = **{NOISE_PEARSON:.4f}** vs local ideal
- The classical baseline (ROC-AUC {CLASSICAL_ROC:.4f}) outperforms the quantum model. **This study does not claim quantum advantage.**
- The goal is hardware fidelity quantification, not performance superiority.

### Figures
![ROC Comparison](figures/roc_comparison.png)
![Architecture](figures/model_architecture.png)
![Circuit](figures/canonical_circuit.png)
![Scatter](figures/ionq_vs_local_scatter.png)
![Shot Stability](figures/shot_stability.png)
"""
write(os.path.join(PKG_DIR, "IONQ_RESEARCH_SUMMARY.md"), summary)

# ── IONQ_RESEARCHER_DRAFT.md ─────────────────────────────────────────────────
draft = f"""Subject: Research Credit Application — Hybrid Quantum Classifier Hardware Fidelity Study
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
| Local ideal (statevector) | {LOCAL_111_ROC:.4f} | 1.000 |
| IonQ ideal simulator | {IONQ_IDEAL_ROC:.4f} | {PEARSON:.4f} |
| IonQ Aria-1 noise model | {IONQ_NOISE_ROC:.4f} | {NOISE_PEARSON:.4f} |

Local full-test reference (76,693 samples, statevector): **ROC-AUC = {LOCAL_IDEAL_ROC:.4f}**

The classical parameter-matched baseline achieves ROC-AUC {CLASSICAL_ROC:.4f}.
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
"""
write(os.path.join(PKG_DIR, "IONQ_RESEARCHER_DRAFT.md"), draft)

# ── Copy all other files ─────────────────────────────────────────────────────
copies = {
    "results/circuit_justification.md": "circuit_justification.md",
    "results/circuit_diagram.txt": "circuit_diagram.txt",
    "results/circuit_equivalence.json": "circuit_equivalence.json",
    "results/parameter_verification.json": "parameter_verification.json",
    "results/shot_stability.json": "shot_stability.json",
    "results/experiment_comparison.csv": "experiment_comparison.csv",
    "results/ionq_sim_results.json": "ionq_sim_results.json",
    "results/all_results.json": "all_results.json",
    "results/reproducibility_manifest.json": "reproducibility_manifest.json",
    "src/canonical_circuit.py": "src/canonical_circuit.py",
    "scripts/run_ionq_integration.py": "scripts/run_ionq_integration.py",
    "scripts/shot_stability.py": "scripts/shot_stability.py",
    "scripts/final_validation.py": "scripts/final_validation.py",
    "scripts/generate_manifest.py": "scripts/generate_manifest.py",
    "scripts/scrub_credentials.py": "scripts/scrub_credentials.py",
    "scripts/generate_figures.py": "scripts/generate_figures.py",
    "checkpoints/canonical_109param_weights.pth": "checkpoints/canonical_109param_weights.pth",
}
for src_rel, dst_rel in copies.items():
    src = os.path.join(ROOT, src_rel)
    dst = os.path.join(PKG_DIR, dst_rel)
    if os.path.exists(src):
        shutil.copy(src, dst)
    else:
        print(f"[WARN] Missing: {src_rel}")

# ── Copy figures ─────────────────────────────────────────────────────────────
for fig_name in ["roc_comparison.png", "ionq_vs_local_scatter.png",
                 "canonical_circuit.png", "shot_stability.png", "model_architecture.png"]:
    src = os.path.join(FIG_SRC, fig_name)
    dst = os.path.join(PKG_DIR, "figures", fig_name)
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"[OK] figures/{fig_name} -> package")
    else:
        print(f"[FAIL] Missing figure: {fig_name}")

# ── Credential scan ──────────────────────────────────────────────────────────
safe = check_no_credentials(PKG_DIR)
if not safe:
    import sys; sys.exit(1)
print("[OK] Credential scan passed.")

# ── ZIP ───────────────────────────────────────────────────────────────────────
zip_base = os.path.join(ROOT, "IONQ_CRISPR_OFFTARGET_FINAL_PACKAGE")
shutil.make_archive(zip_base, "zip", PKG_DIR)
zip_file = zip_base + ".zip"
zip_size = os.path.getsize(zip_file)
zip_hash = get_hash(zip_file)

# Verify figures are inside zip
import zipfile
with zipfile.ZipFile(zip_file) as zf:
    names = zf.namelist()
fig_in_zip = [n for n in names if n.startswith("figures/") and n.endswith(".png")]

with open(os.path.join(ROOT, "IONQ_PACKAGE_SHA256.txt"), "w") as f:
    f.write(zip_hash)

print(f"\n{'='*60}")
print(f"ZIP Path:  {zip_file}")
print(f"ZIP Size:  {zip_size:,} bytes")
print(f"SHA-256:   {zip_hash}")
print(f"\nFigures in ZIP ({len(fig_in_zip)}/5):")
for fn in sorted(fig_in_zip):
    print(f"  {fn}")
