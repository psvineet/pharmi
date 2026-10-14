# PharmI — Trustworthy AI for Pharmaceutical Decision Support

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

> *Protecting cyclodextrin solubility predictions from corrupted data and adversarial attacks.*

**PharmI** is an open-source Python prototype that wraps a machine learning model inside a six-layer security framework. It predicts aqueous solubility of poorly soluble drugs when complexed with cyclodextrins (α, β, γ), while actively defending against SQL injection, data poisoning, adversarial inputs, and manipulated outputs.

Built at the intersection of pharmaceutical sciences and cybersecurity.

---

## The Problem

AI models are increasingly used in pre-formulation to predict drug-cyclodextrin solubility. They save months of lab work. But they carry a critical assumption — that the data fed into them is always clean and correct.

If a dataset is poisoned, an input is manipulated, or an output is tampered with, the model will still confidently return a prediction. Nobody will know it is wrong. That leads to incorrect formulation decisions, wasted resources, and potential downstream clinical consequences.

**PharmI solves this.**

---

## How It Works

PharmI is built around a **Random Forest Regressor** with 300 decision trees trained on molecular descriptor data from 30 real reference drugs (Ibuprofen, Paracetamol, Cyclosporine, Paclitaxel, and 26 others). It takes five molecular descriptors as input and predicts aqueous solubility in mg/mL.

| Input Descriptor | Description |
|---|---|
| logP | Octanol-water partition coefficient — measures hydrophobicity |
| MW | Molecular weight (g/mol) |
| HBD | Hydrogen bond donors |
| HBA | Hydrogen bond acceptors |
| TPSA | Topological polar surface area (Å²) |
| CD Type | Alpha (0), Beta (1), or Gamma (2) cyclodextrin |

Around this core, **six security rules** protect every prediction:

### Before Prediction

| Rule | Name | What It Does | Status |
|---|---|---|---|
| 1 | SQL Injection Protection | Scans inputs for malicious database commands (DROP, UNION, etc.) using regex. Parameterized queries enforced. | `BLOCKED` |
| 2 | SHA-256 Data Integrity | Generates a SHA-256 fingerprint on every database record at insertion. Any tampering causes a hash mismatch. | `QUARANTINED` |
| 3 | Molecular Feature Range Check | Validates that all input descriptors fall within physically realistic bounds for real drugs. | `REJECTED` |

### After Prediction

| Rule | Name | What It Does | Status |
|---|---|---|---|
| 4 | Physical Boundary Check | Confirms predicted solubility falls within 0.001–200 mg/mL. | `QUARANTINED` |
| 5 | Z-Score Anomaly Detection | Flags predictions more than 4 standard deviations from training distribution. | `QUARANTINED` |
| 6 | Model Confidence Score | Measures inter-tree variance across all 300 trees. Outputs High / Medium / Low confidence rating. | `FLAGGED` |

---

## Attack Scenarios Validated

| Attack | Description | Caught By |
|---|---|---|
| A | SQL injection via input field | Rule 1 |
| B | Direct database record modification (data poisoning) | Rule 2 |
| C | Extreme feature values (e.g. logP = 50, MW = 50,000) | Rule 3 |
| D | Post-prediction output multiplication by random multiplier | Rules 4 & 5 |

You can test all four attacks live via the CLI or Web UI, and adjust the data poisoning fraction in real time to observe model degradation and security response.

---

## Model Performance (Clean Data)

| Metric | Value |
|---|---|
| R² | ~0.96 |
| MAE | ~0.4 mg/mL |
| RMSE | ~0.6 mg/mL |
| Training set | 480 records |
| Test set | 120 records |
| Reference drugs | 30 |

---

## Installation

```bash
git clone https://github.com/psvineet/pharmi.git
cd pharmi
pip install flask scikit-learn pandas numpy
```

Dependencies are also auto-installed on first run if missing.

---

## Usage

```bash
# CLI demo only
python pharmI.py --mode cli

# Web UI on http://localhost:5000
python pharmI.py --mode web

# Both CLI demo + Web UI
python pharmI.py --mode both
```

---

## Project Structure

```
pharmi/
│
├── pharmI.py          # Full application — ML model, security layer, CLI, Web UI
├── README.md          # You are here
└── pharmi.db          # SQLite database (auto-created on first run)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Machine Learning | scikit-learn — RandomForestRegressor |
| Data | NumPy, Pandas |
| Database | SQLite with SHA-256 integrity hashing |
| Web Interface | Flask |
| Security | Regex-based SQL scanner, SHA-256, Z-score, CV scoring |
| Language | Python 3.8+ |

---

## Vision

PharmI is currently a working prototype presented at **ICS MOGA 2026**.

Our goal is to develop this into a **B2B cloud platform** — a plug-in security and prediction layer that pharmaceutical companies, research institutions, and formulation labs can integrate directly into their existing AI pipelines. No complex setup. No massive security budget. Just clean, verified, trustworthy predictions — out of the box.

**We are actively looking for collaborators and institutional partners.**

---

## Authors

**Vineet Pratap Singh** — Pharmacy student. Cybersecurity enthusiast.

---

## References

- Loftsson & Brewster (2010) — *Pharmaceutical solubilization of drugs*, J Pharm Sci
- Higuchi & Connors (1965) — *Phase solubility techniques*, Adv Anal Chem Instrum
- Brewster & Loftsson (2007) — *Cyclodextrins as pharmaceutical solubilizers*, Adv Drug Deliv Rev
- Faucci et al. (2004) — *Multivariate analysis of cyclodextrin complexation*, Pharm Res

---

## License

© 2026 Vineet Pratap Singh. Licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

Free for non-commercial use with attribution. **Commercial use requires written permission from the author.**

---

*If this project interests you or you would like to collaborate, feel free to open an issue or reach out directly.*
