import json
import re

notebook_path = "Quantum_CRISPR_Phase1_ipynb_FINAL for submission (1).ipynb"

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        for i, line in enumerate(source):
            if "token" in line.lower() or "api_key" in line.lower() or "sk-" in line:
                # Scrub the line
                source[i] = re.sub(r'["\'][A-Za-z0-9\-_]+["\']', '"REDACTED"', line)
                print(f"Scrubbed line: {line.strip()} -> {source[i].strip()}")

with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
    
print("Notebook scrubbing complete.")
