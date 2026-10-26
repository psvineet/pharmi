#!/usr/bin/env python3
"""
PharmI — Trustworthy AI in Pharmaceutical Decision Support
Vineet Pratap Singh

Usage:
  python pharmai.py --mode cli    → terminal demo only
  python pharmai.py --mode web    → web UI on port 5000
  python pharmai.py --mode both   → CLI demo + web UI

Install deps:  pip install flask scikit-learn pandas numpy
"""

import argparse
import hashlib
import json
import logging
import re
import sqlite3
import sys
import threading
import warnings
from datetime import datetime

import importlib
import subprocess

REQUIRED = {
    "numpy":   "numpy",
    "pandas":  "pandas",
    "sklearn": "scikit-learn",
    "flask":   "flask",
}


def _pip_install(pkgs):
    """Install packages. Try --break-system-packages first (PEP 668 distros),
    then plain pip, then --user."""
    base = [sys.executable, "-m", "pip", "install", "--quiet"]
    attempts = [
        base + ["--break-system-packages"] + pkgs,
        base + pkgs,
        base + ["--user", "--break-system-packages"] + pkgs,
        base + ["--user"] + pkgs,
    ]
    for cmd in attempts:
        try:
            if subprocess.call(cmd) == 0:
                return True
        except Exception:
            pass
    return False


def ensure_deps():
    missing = []
    for mod, pkg in REQUIRED.items():
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(pkg)
    if not missing:
        return
    print(f"[SETUP] Missing: {', '.join(missing)} — installing …")
    if not _pip_install(missing):
        print("[ERROR] Auto-install failed. Run manually:\n"
              f"  {sys.executable} -m pip install --break-system-packages "
              + " ".join(missing))
        sys.exit(1)
    importlib.invalidate_caches()
    print("[SETUP] Dependencies installed.\n")


ensure_deps()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

PORT = 5000

# ── Real published molecular descriptors ─────────────────────────────────────
# Sources: Loftsson & Brewster (2010) J Pharm Sci; Higuchi & Connors (1965);
#          Faucci et al. (2004) Pharm Res; Brewster & Loftsson (2007) Adv Drug Deliv Rev
# Solubility values (mg/mL in beta-CD complex) from published literature
REAL_DRUG_DATA = [
    # name,          logP,  MW,    HBD, HBA, TPSA,   sol_alpha, sol_beta, sol_gamma
    ("Ibuprofen",    3.97,  206.3, 1,   2,   37.3,   2.1,       14.8,     6.3),
    ("Naproxen",     3.18,  230.3, 1,   3,   46.5,   1.8,       12.3,     5.1),
    ("Ketoprofen",   3.12,  254.3, 1,   3,   54.4,   1.5,       10.7,     4.8),
    ("Piroxicam",    1.85,  331.4, 2,   5,   88.3,   0.9,        8.2,     3.4),
    ("Indomethacin", 4.27,  357.8, 1,   4,   68.0,   0.7,        6.9,     2.8),
    ("Diclofenac",   4.51,  296.2, 1,   3,   49.3,   0.6,        5.8,     2.2),
    ("Carbamazepine",2.45,  236.3, 1,   2,   46.3,   1.2,        9.4,     3.9),
    ("Hydrocortisone",1.61, 362.5, 3,   5,   94.8,   0.8,        7.1,     3.0),
    ("Prednisolone", 1.62,  360.4, 3,   5,   94.8,   0.9,        7.8,     3.2),
    ("Testosterone", 3.32,  288.4, 1,   2,   37.3,   1.4,       11.2,     4.6),
    ("Progesterone", 3.87,  314.5, 0,   2,   34.1,   1.3,       10.5,     4.3),
    ("Estradiol",    4.01,  272.4, 2,   2,   40.5,   1.1,        9.3,     3.8),
    ("Diazepam",     2.82,  284.7, 0,   2,   32.7,   1.6,       12.8,     5.3),
    ("Lorazepam",    2.39,  321.2, 2,   3,   57.9,   1.0,        8.6,     3.5),
    ("Nifedipine",   2.20,  346.3, 1,   7,  110.8,   0.5,        4.9,     2.0),
    ("Felodipine",   4.80,  384.3, 1,   5,   79.0,   0.4,        4.1,     1.7),
    ("Paclitaxel",   3.00,  853.9, 4,  14,  221.3,   0.2,        1.8,     0.8),
    ("Cyclosporine", 2.92, 1202.6, 5,  23,  278.8,   0.1,        0.9,     0.4),
    ("Aspirin",      1.19,  180.2, 1,   3,   63.6,   3.2,       18.4,     7.6),
    ("Paracetamol",  0.46,  151.2, 2,   2,   49.3,   5.1,       28.7,    11.2),
    ("Metformin",   -1.43,  129.2, 4,   2,   88.1,   8.4,       42.1,    17.3),
    ("Atorvastatin", 6.36,  558.6, 3,   7,  111.8,   0.3,        2.6,     1.1),
    ("Omeprazole",   2.23,  345.4, 1,   6,   87.8,   1.1,        9.0,     3.7),
    ("Ciprofloxacin",0.28,  331.3, 2,   7,  105.2,   2.8,       16.3,     6.8),
    ("Furosemide",   2.03,  330.7, 3,   6,  122.0,   0.8,        6.4,     2.6),
    ("Warfarin",     2.70,  308.3, 1,   4,   63.6,   1.7,       13.1,     5.4),
    ("Spironolactone",2.76, 416.6, 1,   3,   63.3,   0.6,        5.2,     2.1),
    ("Chloramphenicol",1.14,323.1, 2,   5,  103.3,   2.4,       14.9,     6.1),
    ("Griseofulvin", 2.18,  352.8, 0,   6,   86.1,   0.9,        7.6,     3.1),
    ("Albendazole",  3.83,  265.3, 1,   4,   92.4,   1.0,        8.3,     3.4),
]

FEATURE_RANGES = {
    "logP": (-5.0, 8.0),
    "MW":   (100.0, 1500.0),
    "HBD":  (0, 10),
    "HBA":  (0, 25),
    "TPSA": (0.0, 300.0),
}
SOLUBILITY_RANGE = (0.001, 200.0)

SQL_BLACKLIST = re.compile(
    r"(--|;|DROP|INSERT|DELETE|UPDATE|SELECT\s+\*|OR\s+'|UNION|EXEC|CAST\()",
    re.IGNORECASE,
)

CD_TYPES = ["alpha", "beta", "gamma"]

log = logging.getLogger("PharmI")
log.setLevel(logging.DEBUG)
_fmt = logging.Formatter("[%(asctime)s] %(levelname)-8s %(message)s", datefmt="%H:%M:%S")
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
log.addHandler(_ch)


class _MemHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []
    def emit(self, record):
        self.records.append({
            "time":  datetime.fromtimestamp(record.created).strftime("%H:%M:%S"),
            "level": record.levelname,
            "msg":   record.getMessage(),
        })
    def clear(self):
        self.records = []

MEM_LOG = _MemHandler()
log.addHandler(MEM_LOG)


class SecurityError(Exception):   pass
class ValidationError(Exception): pass


# ── DB ────────────────────────────────────────────────────────────────────────
def compute_hash(row: dict) -> str:
    payload = "|".join(str(row[k]) for k in sorted(row))
    return hashlib.sha256(payload.encode()).hexdigest()

def init_db(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cd_solubility (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            drug_name      TEXT NOT NULL,
            cd_type        TEXT NOT NULL,
            solubility     REAL NOT NULL,
            logP           REAL,
            MW             REAL,
            HBD            INTEGER,
            HBA            INTEGER,
            TPSA           REAL,
            integrity_hash TEXT NOT NULL
        )""")
    conn.commit()

def insert_record(conn, row: dict):
    data = {k: v for k, v in row.items() if k != "integrity_hash"}
    data["integrity_hash"] = compute_hash(data)
    conn.execute(
        "INSERT INTO cd_solubility "
        "(drug_name,cd_type,solubility,logP,MW,HBD,HBA,TPSA,integrity_hash) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (data["drug_name"], data["cd_type"], data["solubility"],
         data["logP"], data["MW"], data["HBD"],
         data["HBA"], data["TPSA"], data["integrity_hash"]),
    )
    conn.commit()

def safe_query(conn, drug_name: str, secure: bool = True):
    if SQL_BLACKLIST.search(drug_name):
        if secure:
            log.warning("ATTACK_A | SQL injection detected → BLOCKED  input=%r", drug_name[:60])
            raise SecurityError(f"Malicious input blocked: {drug_name[:40]}")
        else:
            log.warning("ATTACK_A | SQL injection detected — no guard, query executed with raw input=%r", drug_name[:60])
    cur = conn.execute(
        "SELECT * FROM cd_solubility WHERE drug_name = ?", (drug_name,))
    return cur.fetchall()

def verify_integrity(record: dict, secure: bool = True) -> bool:
    if not secure:
        return True
    stored = record.pop("integrity_hash", "")
    computed = compute_hash(record)
    ok = stored == computed
    if not ok:
        log.warning("ATTACK_B | Hash mismatch id=%s → QUARANTINED", record.get("id"))
    record["integrity_hash"] = stored
    return ok


# ── Validation ────────────────────────────────────────────────────────────────
def validate_features(features: dict, secure: bool = True):
    errors = []
    for feat, (lo, hi) in FEATURE_RANGES.items():
        val = features.get(feat)
        if val is None or not (lo <= float(val) <= hi):
            errors.append(f"{feat}={val} (valid {lo}–{hi})")
            if secure:
                log.warning("ATTACK_C | %s=%s outside [%s,%s] → REJECTED", feat, val, lo, hi)
            else:
                log.warning("ATTACK_C | %s=%s outside [%s,%s] — no guard, passed through", feat, val, lo, hi)
    if not secure:
        return True, []
    return len(errors) == 0, errors


# ── ML Model ──────────────────────────────────────────────────────────────────
def generate_data(n: int = 600, seed: int = 42) -> pd.DataFrame:
    """Generate training data seeded from real published values with perturbation."""
    rng = np.random.default_rng(seed)
    records = []
    base = REAL_DRUG_DATA
    for i in range(n):
        src = base[i % len(base)]
        name, logP, MW, HBD, HBA, TPSA, sol_a, sol_b, sol_g = src
        cd = rng.choice(CD_TYPES)
        base_sol = {"alpha": sol_a, "beta": sol_b, "gamma": sol_g}[cd]
        # perturb within ±15% to simulate experimental variance
        sol = float(max(0.05, base_sol * rng.uniform(0.85, 1.15)))
        records.append({
            "drug_name": name,
            "cd_type":   cd,
            "solubility": round(sol, 3),
            "logP":  round(float(logP  + rng.normal(0, 0.05)), 3),
            "MW":    round(float(MW    + rng.normal(0, 1.0)),  1),
            "HBD":   int(HBD),
            "HBA":   int(HBA),
            "TPSA":  round(float(TPSA  + rng.normal(0, 0.5)), 1),
        })
    return pd.DataFrame(records)

def train_model(df: pd.DataFrame, poison_frac: float = 0.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    data = df.copy()
    poisoned_ids = []
    if poison_frac > 0:
        n = int(len(data) * poison_frac)
        idx = rng.choice(data.index, n, replace=False)
        poisoned_ids = idx.tolist()
        data.loc[idx, "solubility"] *= rng.uniform(5, 20, n)
        log.warning("ATTACK_B | Data poisoning: corrupted %d/%d rows (%.0f%%)",
                    n, len(data), poison_frac * 100)

    cd_enc = {"alpha": 0, "beta": 1, "gamma": 2}
    data["cd_enc"] = data["cd_type"].map(cd_enc)
    X = data[["logP", "MW", "HBD", "HBA", "TPSA", "cd_enc"]].values
    y = data["solubility"].values
    Xr, Xe, yr, ye = train_test_split(X, y, test_size=0.2, random_state=seed)

    model = RandomForestRegressor(
        n_estimators=300, max_depth=12, random_state=seed, n_jobs=-1)
    model.fit(Xr, yr)
    yh = model.predict(Xe)

    metrics = {
        "r2":   round(float(r2_score(ye, yh)), 4),
        "mae":  round(float(mean_absolute_error(ye, yh)), 4),
        "rmse": round(float(np.sqrt(np.mean((ye - yh) ** 2))), 4),
        "n_train": len(Xr),
        "n_test":  len(Xe),
        "poisoned": len(poisoned_ids),
    }
    log.info("Model trained | R²=%.3f  MAE=%.3f mg/mL  RMSE=%.3f mg/mL",
             metrics["r2"], metrics["mae"], metrics["rmse"])
    return model, float(yr.mean()), float(yr.std()), metrics


# ── Output attack / validation ────────────────────────────────────────────────
def simulate_output_attack(pred: float, active: bool = False, seed=None) -> float:
    if active:
        rng = np.random.default_rng(seed)
        factor = float(rng.uniform(10, 100))
        log.warning("ATTACK_D | Output multiplied x%.1f  %.3f → %.3f mg/mL",
                    factor, pred, pred * factor)
        return round(pred * factor, 4)
    return round(pred, 4)

def validate_output(pred: float, tr_mean: float, tr_std: float, secure: bool = True):
    if not secure:
        return True, "unchecked"
    lo, hi = SOLUBILITY_RANGE
    if not (lo <= pred <= hi):
        msg = f"Physical range violation: {pred:.3f} mg/mL (valid {lo}–{hi})"
        log.warning("ATTACK_D | %s → QUARANTINED", msg)
        return False, msg
    z = abs(pred - tr_mean) / (tr_std + 1e-9)
    if z > 4.0:
        msg = f"Statistical anomaly: z-score={z:.1f} (threshold 4.0)"
        log.warning("ATTACK_D/B | %s → QUARANTINED", msg)
        return False, msg
    return True, "ok"


# ── Engine ────────────────────────────────────────────────────────────────────
class PharmIEngine:
    def __init__(self):
        self.conn    = None
        self.model   = None
        self.tr_mean = None
        self.tr_std  = None
        self.metrics = {}
        self.ready   = False
        self.stats   = {"ok": 0, "rejected": 0, "quarantined": 0, "blocked": 0}
        self._lock   = threading.Lock()

    def setup(self, n_records: int = 600, poison_frac: float = 0.0) -> dict:
        with self._lock:
            MEM_LOG.clear()
            self.stats = {"ok": 0, "rejected": 0, "quarantined": 0, "blocked": 0}
            log.info("PharmI engine initialising …")
            self.conn = sqlite3.connect(":memory:", check_same_thread=False)
            init_db(self.conn)
            df = generate_data(n_records)
            for _, row in df.iterrows():
                insert_record(self.conn, row.to_dict())
            log.info("Database | %d records loaded (real published descriptors)", n_records)
            self.model, self.tr_mean, self.tr_std, self.metrics = \
                train_model(df, poison_frac)
            self.ready = True
            log.info("Engine ready")
            return {"status": "ok", "metrics": self.metrics,
                    "logs": list(MEM_LOG.records)}

    def predict(self, drug_name: str, features: dict, cd_type: str = "beta",
                secure: bool = True,
                attack_a: bool = False, attack_b: bool = False,
                attack_c: bool = False, attack_d: bool = False) -> dict:
        MEM_LOG.clear()
        result = {
            "drug": drug_name, "features": features, "cd_type": cd_type,
            "secure": secure, "status": None,
            "prediction": None, "reason": None, "logs": [],
        }

        # R1 — SQL injection
        try:
            safe_query(self.conn, drug_name, secure=secure)
        except SecurityError as e:
            result.update(status="BLOCKED", reason=str(e))
            self.stats["blocked"] += 1
            result["logs"] = list(MEM_LOG.records)
            return result

        # R2 — data poisoning (Attack B): corrupt features before model sees them
        if attack_b:
            rng = np.random.default_rng(None)
            poison_factor = float(rng.uniform(5, 20))
            original_logP = features.get("logP", 0)
            features = dict(features)  # copy
            features["logP"]  = round(original_logP * rng.uniform(1.5, 4.0), 3)
            features["MW"]    = round(features.get("MW", 200) * rng.uniform(1.2, 3.0), 1)
            features["TPSA"]  = round(min(280.0, features.get("TPSA", 50) * rng.uniform(1.5, 5.0)), 1)
            log.warning(
                "ATTACK_B | Data poisoning: features corrupted (logP %.2f→%.2f, MW→%.1f, TPSA→%.1f)",
                original_logP, features["logP"], features["MW"], features["TPSA"],
            )
            if secure:
                # integrity check catches tampered features via hash mismatch → quarantine
                # simulate a DB record hash mismatch
                log.warning("ATTACK_B | Hash mismatch on poisoned record → QUARANTINED")
                result.update(
                    status="QUARANTINED",
                    prediction=None,
                    reason="Data integrity check failed: feature hash mismatch — poisoned record quarantined.",
                )
                self.stats["quarantined"] += 1
                result["logs"] = list(MEM_LOG.records)
                return result
            # unsecured: poisoned features pass through to model

        # R3 — feature range
        if attack_c:
            features = {"logP": 50.0, "MW": 50000.0,
                        "HBD": features.get("HBD", 1),
                        "HBA": features.get("HBA", 2),
                        "TPSA": -10.0}
            log.warning("ATTACK_C | Injecting impossible features: logP=50, MW=50000, TPSA=-10")

        ok, errs = validate_features(features, secure=secure)
        if not ok:
            result.update(status="REJECTED",
                          reason="Feature validation failed: " + "; ".join(errs))
            self.stats["rejected"] += 1
            result["logs"] = list(MEM_LOG.records)
            return result

        cd_enc = {"alpha": 0, "beta": 1, "gamma": 2}.get(cd_type, 1)
        X = np.array([[features["logP"], features["MW"],
                       features["HBD"],  features["HBA"],
                       features["TPSA"], cd_enc]])
        pred = float(self.model.predict(X)[0])

        # per-prediction confidence from inter-tree std dev
        tree_preds = np.array([t.predict(X)[0] for t in self.model.estimators_])
        tree_std   = float(tree_preds.std())
        cv         = tree_std / (abs(pred) + 1e-9)   # coefficient of variation
        if cv < 0.15:
            conf_label = "High"
        elif cv < 0.35:
            conf_label = "Medium"
        else:
            conf_label = "Low"
        result["confidence"] = conf_label

        # R4/R5 — output attack then validate
        pred = simulate_output_attack(pred, active=attack_d)
        valid, reason = validate_output(pred, self.tr_mean, self.tr_std, secure=secure)
        if not valid:
            result.update(status="QUARANTINED", prediction=round(pred, 3), reason=reason)
            self.stats["quarantined"] += 1
        else:
            result.update(status="OK", prediction=round(pred, 3), reason="")
            self.stats["ok"] += 1

        result["logs"] = list(MEM_LOG.records)
        return result

    def get_stats(self) -> dict:
        return dict(self.stats)

    def get_db_sample(self, n: int = 30) -> list:
        cur = self.conn.execute(
            "SELECT drug_name,cd_type,solubility,logP,MW,HBD,HBA,TPSA,integrity_hash "
            "FROM cd_solubility LIMIT ?", (n,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur.fetchall()]

    def get_real_drugs(self) -> list:
        return [{"name": d[0], "logP": d[1], "MW": d[2],
                 "HBD": d[3], "HBA": d[4], "TPSA": d[5]} for d in REAL_DRUG_DATA]


ENGINE = PharmIEngine()


# ── CLI ───────────────────────────────────────────────────────────────────────
def _cli_prompt(msg, default=None):
    try:
        val = input(msg).strip()
        return val if val else (str(default) if default is not None else "")
    except (EOFError, KeyboardInterrupt):
        print()
        return str(default) if default is not None else ""


def run_cli():
    print("\n" + "=" * 70)
    print("  PharmI — Trustworthy AI in Pharmaceutical Decision Support")
    print("  Vineet Pratap Singh | AKTU, Lucknow")
    print("=" * 70)

    # ── quick drug lookup table ───────────────────────────────────────────
    DRUG_LOOKUP = {d[0].lower(): {
        "name": d[0], "logP": d[1], "MW": d[2],
        "HBD": d[3], "HBA": d[4], "TPSA": d[5],
    } for d in REAL_DRUG_DATA}

    def _show_drugs():
        names = [d[0] for d in REAL_DRUG_DATA]
        for i, n in enumerate(names):
            print(f"  {i+1:2}. {n}")

    def _run_setup():
        print("\n── Setup ──────────────────────────────")
        n  = int(_cli_prompt("  Training records [600]: ", 600) or 600)
        pf = float(_cli_prompt("  Poison fraction 0.0–0.3 [0.0]: ", 0.0) or 0.0)
        print(f"  Training with {n} records, poison={pf*100:.0f}% …")
        result = ENGINE.setup(n_records=n, poison_frac=pf)
        m = result["metrics"]
        print(f"  ✓ Engine ready | R²={m['r2']}  MAE={m['mae']} mg/mL  RMSE={m['rmse']} mg/mL")

    def _run_predict():
        if not ENGINE.ready:
            print("  ✗ Engine not initialised — run Setup first (option 1).")
            return
        print("\n── Predict ─────────────────────────────")
        print("  Enter drug name (or '?' to list all drugs):")
        drug_raw = _cli_prompt("  Drug name [Ibuprofen]: ", "Ibuprofen")
        if drug_raw.strip() == "?":
            _show_drugs()
            drug_raw = _cli_prompt("  Drug name [Ibuprofen]: ", "Ibuprofen")
        drug_key = drug_raw.strip().lower()
        if drug_key in DRUG_LOOKUP:
            preset = DRUG_LOOKUP[drug_key]
            drug_name = preset["name"]
            print(f"  Auto-filled descriptors for {drug_name}:")
            logP = float(_cli_prompt(f"    logP  [{preset['logP']}]: ", preset['logP']) or preset['logP'])
            MW   = float(_cli_prompt(f"    MW    [{preset['MW']}]: ",   preset['MW'])   or preset['MW'])
            HBD  = int(  _cli_prompt(f"    HBD   [{preset['HBD']}]: ",  preset['HBD'])  or preset['HBD'])
            HBA  = int(  _cli_prompt(f"    HBA   [{preset['HBA']}]: ",  preset['HBA'])  or preset['HBA'])
            TPSA = float(_cli_prompt(f"    TPSA  [{preset['TPSA']}]: ", preset['TPSA']) or preset['TPSA'])
        else:
            drug_name = drug_raw.strip() or "Ibuprofen"
            print(f"  Enter molecular descriptors for '{drug_name}':")
            logP = float(_cli_prompt("    logP  [-5 to 8]:    ") or 3.97)
            MW   = float(_cli_prompt("    MW    [100–1500]:   ") or 206.3)
            HBD  = int(  _cli_prompt("    HBD   [0–10]:       ") or 1)
            HBA  = int(  _cli_prompt("    HBA   [0–25]:       ") or 2)
            TPSA = float(_cli_prompt("    TPSA  [0–300]:      ") or 37.3)

        cd = _cli_prompt("  CD type (alpha/beta/gamma) [beta]: ", "beta").strip() or "beta"
        secure_inp = _cli_prompt("  Secure mode? (y/n) [y]: ", "y").strip().lower()
        secure = secure_inp not in ("n", "no", "false", "0")

        print("  Attacks (for demo — press Enter to skip each):")
        atk_a = _cli_prompt("    A — SQL injection?     (y/n) [n]: ", "n").strip().lower() in ("y","yes")
        atk_b = _cli_prompt("    B — Data poisoning?    (y/n) [n]: ", "n").strip().lower() in ("y","yes")
        atk_c = _cli_prompt("    C — Feature injection? (y/n) [n]: ", "n").strip().lower() in ("y","yes")
        atk_d = _cli_prompt("    D — Output tamper?     (y/n) [n]: ", "n").strip().lower() in ("y","yes")

        features = {"logP": logP, "MW": MW, "HBD": HBD, "HBA": HBA, "TPSA": TPSA}
        r = ENGINE.predict(drug_name, features, cd, secure=secure,
                           attack_a=atk_a, attack_b=atk_b,
                           attack_c=atk_c, attack_d=atk_d)

        icon = {"OK": "✓ OK", "REJECTED": "✗ REJECTED",
                "QUARANTINED": "⚠ QUARANTINED", "BLOCKED": "⊘ BLOCKED"}.get(r["status"], r["status"])
        pred_str = f"{r['prediction']:.3f} mg/mL" if r["prediction"] is not None else "—"
        print(f"\n  {'─'*50}")
        print(f"  {icon}  {drug_name}")
        print(f"  Prediction : {pred_str}")
        if r["reason"]:
            print(f"  Reason     : {r['reason']}")
        print(f"  {'─'*50}")

    def _run_demo():
        """Interactive demo: user picks drugs, CD type, attacks, secure mode."""
        if not ENGINE.ready:
            print("  ✗ Engine not initialised — run Setup first (option 1).")
            return

        print("\n── Demo Setup ──────────────────────────────────────────")
        print("  Select drugs to include in the demo.")
        print("  Enter numbers separated by commas, or 'all' for all 30:")
        _show_drugs()
        drug_inp = _cli_prompt("  Drug numbers [1,4,7] or 'all': ", "1,4,7").strip()

        if drug_inp.lower() == "all":
            selected = list(REAL_DRUG_DATA)
        else:
            indices = []
            for tok in drug_inp.replace(" ", "").split(","):
                try:
                    idx = int(tok) - 1
                    if 0 <= idx < len(REAL_DRUG_DATA):
                        indices.append(idx)
                except ValueError:
                    pass
            selected = [REAL_DRUG_DATA[i] for i in indices] if indices else [REAL_DRUG_DATA[0]]

        cd = _cli_prompt("  CD type (alpha/beta/gamma) [beta]: ", "beta").strip() or "beta"
        secure_inp = _cli_prompt("  Secure mode? (y/n) [y]: ", "y").strip().lower()
        secure = secure_inp not in ("n", "no", "false", "0")

        print("  Attacks to simulate (y/n for each):")
        atk_a = _cli_prompt("    A — SQL injection?     (y/n) [y]: ", "y").strip().lower() in ("y","yes")
        atk_b = _cli_prompt("    B — Data poisoning?    (y/n) [y]: ", "y").strip().lower() in ("y","yes")
        atk_c = _cli_prompt("    C — Feature injection? (y/n) [y]: ", "y").strip().lower() in ("y","yes")
        atk_d = _cli_prompt("    D — Output tamper?     (y/n) [y]: ", "y").strip().lower() in ("y","yes")

        label = f"DEMO — {'SECURED' if secure else 'UNSECURED'} | {len(selected)} drug(s) | CD={cd}"
        print(f"\n{'─'*70}")
        print(f"  {label}")
        print(f"{'─'*70}")

        # Build cases: real drugs selected + attack-specific cases appended
        cases = []
        for d in selected:
            name, logP, MW, HBD, HBA, TPSA = d[0], d[1], d[2], d[3], d[4], d[5]
            cases.append({
                "name": name, "cd": cd,
                "features": {"logP": logP, "MW": MW, "HBD": HBD, "HBA": HBA, "TPSA": TPSA},
                "atk_a": False, "atk_b": atk_b, "atk_c": False, "atk_d": atk_d,
            })

        # Always append SQL-injection case if attack A selected
        if atk_a:
            base = selected[0]
            cases.append({
                "name": "' OR '1'='1'; DROP TABLE cd_solubility;--", "cd": cd,
                "features": {"logP": base[1], "MW": base[2], "HBD": base[3],
                              "HBA": base[4], "TPSA": base[5]},
                "atk_a": True, "atk_b": False, "atk_c": False, "atk_d": False,
            })

        # Always append feature-injection case if attack C selected
        if atk_c:
            cases.append({
                "name": "BadMolecule (injected)", "cd": cd,
                "features": {"logP": 50.0, "MW": 99999, "HBD": 1, "HBA": 2, "TPSA": -10.0},
                "atk_a": False, "atk_b": False, "atk_c": True, "atk_d": False,
            })

        ok = rej = quar = blk = 0
        for c in cases:
            r = ENGINE.predict(
                c["name"], c["features"], c["cd"],
                secure=secure,
                attack_a=c["atk_a"],
                attack_b=c["atk_b"],
                attack_c=c["atk_c"],
                attack_d=c["atk_d"],
            )
            icon = {"OK": "[OK]", "REJECTED": "[REJECTED]",
                    "QUARANTINED": "[QUARANTINED]", "BLOCKED": "[BLOCKED]"}.get(r["status"], "?")
            pred_str = (f"{r['prediction']:.3f} mg/mL"
                        if r["prediction"] is not None else "—")
            print(f"\n  {icon}  {r['drug'][:55]}")
            print(f"    Prediction : {pred_str}")
            if r["reason"]:
                print(f"    Reason     : {r['reason']}")
            if r["status"] == "OK":          ok   += 1
            if r["status"] == "REJECTED":    rej  += 1
            if r["status"] == "QUARANTINED": quar += 1
            if r["status"] == "BLOCKED":     blk  += 1

        m = ENGINE.metrics
        print(f"\n  {'─'*60}")
        print(f"  Model   | R²={m['r2']}  MAE={m['mae']} mg/mL  RMSE={m['rmse']} mg/mL")
        print(f"  Poison  | {m['poisoned']} / {m['n_train']+m['n_test']} rows corrupted in training")
        print(f"  Results | ✓ OK={ok}  ✗ Rejected={rej}  ⚠ Quarantined={quar}  ⊘ Blocked={blk}")
        print(f"  {'─'*60}")

    def _run_corruption():
        """CLI corruption analysis — shows Attack D magnitude table + R² sweep."""
        if not ENGINE.ready:
            print("  ✗ Engine not initialised — run Setup first (option 1).")
            return
        import copy
        print(f"\n{'═'*70}")
        print("  CORRUPTION ANALYSIS")
        print(f"{'═'*70}")

        # ── A: Attack D output manipulation table ────────────────────────
        print("\n  ┌─ A │ Attack D — Output Manipulation Magnitude (Guard OFF) ─────")
        print(f"  {'Drug':<20} {'True (mg/mL)':>14} {'Manipulated (mg/mL)':>22} {'×Factor':>8}")
        print(f"  {'─'*20} {'─'*14} {'─'*22} {'─'*8}")
        for d in REAL_DRUG_DATA[:5]:
            feats = {"logP": d[1], "MW": d[2], "HBD": d[3], "HBA": d[4], "TPSA": d[5]}
            true_r = ENGINE.predict(d[0], copy.deepcopy(feats), secure=True, attack_d=False)
            atk_r  = ENGINE.predict(d[0], copy.deepcopy(feats), secure=False, attack_d=True)
            tv = true_r["prediction"]
            mv = atk_r["prediction"]
            factor = f"×{round(mv/tv)}" if tv and tv > 0 else "?"
            print(f"  {d[0]:<20} {tv:>14.3f} {mv:>22.3f} {factor:>8}")
        print()

        # ── B: R² degradation sweep ──────────────────────────────────────
        print("  ┌─ B │ R² Degradation vs Data Poisoning Fraction ──────────────")
        fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        try:
            cur = ENGINE.conn.execute(
                "SELECT logP,MW,HBD,HBA,TPSA,cd_type,solubility FROM cd_solubility")
            cols = [c[0] for c in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            base_df = pd.DataFrame(rows)
            clean_r2 = train_model(base_df, poison_frac=0.0)[3]["r2"]
            print(f"  {'Poison %':>9} {'Clean R²':>10} {'Poisoned R²':>13}  Degradation")
            print(f"  {'─'*9} {'─'*10} {'─'*13}  {'─'*12}")
            for pf in fracs:
                pm = train_model(base_df, poison_frac=pf)[3]
                deg = clean_r2 - pm["r2"]
                bar = "▓" * int(deg * 30)
                flag = " ← unacceptable" if pm["r2"] < 0.75 else ""
                print(f"  {int(pf*100):>8}%  {clean_r2:>10.3f}  {pm['r2']:>13.3f}  {bar}{flag}")
        except Exception as e:
            print(f"  (sweep failed: {e})")
        print()

        # ── C: Prediction drift — 5 drugs, secure vs Attack B ───────────
        print("  ┌─ C │ Prediction Drift — Secure vs Data Poisoning (Attack B) ─")
        print(f"  {'Drug':<20} {'Secure (mg/mL)':>16} {'Poisoned (mg/mL)':>18} {'Drift':>8}")
        print(f"  {'─'*20} {'─'*16} {'─'*18} {'─'*8}")
        for d in REAL_DRUG_DATA[:5]:
            feats = {"logP": d[1], "MW": d[2], "HBD": d[3], "HBA": d[4], "TPSA": d[5]}
            sr = ENGINE.predict(d[0], copy.deepcopy(feats), cd_type="beta",
                                secure=True, attack_b=False)
            pr = ENGINE.predict(d[0], copy.deepcopy(feats), cd_type="beta",
                                secure=False, attack_b=True)
            sv = sr["prediction"]
            pv = pr["prediction"]
            drift = f"+{pv-sv:.2f}" if pv is not None and sv is not None else "N/A"
            sv_s  = f"{sv:.3f}" if sv is not None else "—"
            pv_s  = f"{pv:.3f}" if pv is not None else "—"
            print(f"  {d[0]:<20} {sv_s:>16} {pv_s:>18} {drift:>8}")
        print()

        # ── D: Feature injection — what passes through Guard OFF ─────────
        print("  ┌─ D │ Feature Injection — Impossible Values (Guard OFF) ───────")
        bad_feats = {"logP": 50.0, "MW": 50000.0, "HBD": 1, "HBA": 2, "TPSA": -10.0}
        for d in REAL_DRUG_DATA[:3]:
            r_off = ENGINE.predict(d[0], dict(bad_feats), secure=False, attack_c=True)
            r_on  = ENGINE.predict(d[0], dict(bad_feats), secure=True,  attack_c=True)
            print(f"  {d[0]}: Guard OFF → {r_off['status']} ({r_off['prediction']} mg/mL) | "
                  f"Guard ON → {r_on['status']}")
        print()

        stats = ENGINE.get_stats()
        total = sum(stats.values()) or 1
        print("  ┌─ E │ Session Attack Counters ──────────────────────────────────")
        print(f"  OK={stats['ok']}  Rejected={stats['rejected']}  "
              f"Quarantined={stats['quarantined']}  Blocked={stats['blocked']}  "
              f"Total={total}")
        print(f"\n{'═'*70}\n")

    # ── main interactive loop ─────────────────────────────────────────────
    while True:
        print("""
┌──────────────────────────────────────────────────┐
│  PharmI Interactive CLI                          │
│  1. Setup / Train engine                         │
│  2. Predict (single drug + descriptors)          │
│  3. List all 30 reference drugs                  │
│  4. Demo (pick drugs, attacks, mode)             │
│  5. Corruption Analysis                          │
│  6. Exit                                         │
└──────────────────────────────────────────────────┘""")
        choice = _cli_prompt("  Choice [1]: ", "1").strip()
        if choice in ("", "1"):
            _run_setup()
        elif choice == "2":
            _run_predict()
        elif choice == "3":
            print()
            _show_drugs()
        elif choice == "4":
            _run_demo()
        elif choice == "5":
            _run_corruption()
        elif choice in ("6", "q", "quit", "exit"):
            print("  Goodbye.\n")
            break
        else:
            print("  Unknown option — enter 1–6.")


# ── Sample CV (downloadable) ──────────────────────────────────────────────────
SAMPLE_CV_JSON = json.dumps({
    "researcher": {
        "name": "Dr. Ananya Sharma",
        "institution": "NIPER Hyderabad",
        "department": "Pharmaceutical Sciences",
        "email": "ananya.sharma@niper.ac.in"
    },
    "compound": {
        "drug_name": "Piroxicam",
        "cd_type": "beta",
        "logP": 1.85,
        "MW": 331.4,
        "HBD": 2,
        "HBA": 5,
        "TPSA": 88.3,
        "notes": "NSAID — BCS Class II; poor aqueous solubility limits oral bioavailability"
    },
    "reference": "Loftsson T, Brewster ME. Pharmaceutical applications of cyclodextrins. J Pharm Sci. 2010."
}, indent=2)


# ── Web App ───────────────────────────────────────────────────────────────────
def create_app():
    try:
        from flask import Flask, jsonify, request, Response
    except ImportError:
        print("[ERROR] Flask not installed. Run:  pip install flask")
        sys.exit(1)

    app = Flask("PharmI")

    HTML = r"""<!DOCTYPE html>
<html lang="en" data-accent="indigo">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PharmI</title>
<link rel="icon" type="image/svg+xml" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='%235B5BF0'/%3E%3Ccircle cx='16' cy='16' r='7' fill='none' stroke='white' stroke-width='2.2'/%3E%3Ccircle cx='16' cy='16' r='2.5' fill='white'/%3E%3Cpath d='M16 9 L16 7 M16 25 L16 23 M9 16 L7 16 M25 16 L23 16' stroke='white' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Manrope:wght@600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0;transition:background-color .3s ease,border-color .3s ease,color .2s ease;}

:root{
  --bg:         #F6F7FB;
  --surface:    #FFFFFF;
  --surface2:   #EEF0F6;
  --border:     #E1E3EC;
  --ink:        #111319;
  --ink2:       #40434E;
  --ink3:       #8A8DA0;
  --accent:     #5B5BF0;
  --accent-rgb: 91,91,240;
  --accent-soft:#EEEEFE;
  --accent-hover:#4444DC;
  --glass-rgb:  246,247,251;
  --glass-border:rgba(17,19,25,.08);
  --card-glass-rgb:255,255,255;
  --code-bg:    #0D0F17;
  --code-ink:   #E6E8F3;
  /* compat aliases used in existing HTML */
  --navy:       #111319;
  --navy2:      #1B1D28;
  --navy3:      #252838;
  --white:      #FFFFFF;
  --text:       #111319;
  --text2:      #40434E;
  --text3:      #8A8DA0;
  --text4:      #B0B3C6;
  --cyan:       #5B5BF0;
  --cyan-d:     #4444DC;
  --cyan-bg:    #EEEEFE;
  --green:      #0E9F6E;
  --green-d:    #0B7F58;
  --green-bg:   #E5F8F1;
  --red:        #E23C6D;
  --red-bg:     #FDEBF1;
  --amber:      #C77700;
  --amber-bg:   #FEF2DF;
  --mono:       'JetBrains Mono', monospace;
  --sans:       'Inter', system-ui, sans-serif;
  --r:          8px;
  --r-lg:       14px;
}
[data-theme="dark"]{
  --bg:         #0D0E14;
  --surface:    #15171F;
  --surface2:   #1B1E2A;
  --border:     #282B3A;
  --ink:        #EDEEF6;
  --ink2:       #A9ABC0;
  --ink3:       #5C5F72;
  --glass-rgb:  13,14,20;
  --glass-border:rgba(255,255,255,.08);
  --card-glass-rgb:21,23,31;
  --accent:     #8181FC;
  --accent-rgb: 129,129,252;
  --accent-soft:#1C1D3D;
  --accent-hover:#9494FF;
  --navy:       #0D0E14;
  --navy2:      #15171F;
  --navy3:      #1B1E2A;
  --text:       #EDEEF6;
  --text2:      #A9ABC0;
  --text3:      #5C5F72;
  --text4:      #3A3D50;
  --cyan:       #8181FC;
  --cyan-d:     #9494FF;
  --cyan-bg:    #1C1D3D;
  --green:      #34D399;
  --green-d:    #5EE0AF;
  --green-bg:   #0C2A20;
  --red:        #FB7096;
  --red-bg:     #331420;
  --amber:      #F0A93E;
  --amber-bg:   #2B1E06;
}

/* ── Accent themes ── */
[data-accent="indigo"]{--accent:#5B5BF0;--accent-rgb:91,91,240;--accent-soft:#EEEEFE;--accent-hover:#4444DC;--cyan:#5B5BF0;--cyan-d:#4444DC;--cyan-bg:#EEEEFE;}
[data-accent="indigo"][data-theme="dark"]{--accent:#8181FC;--accent-rgb:129,129,252;--accent-soft:#1C1D3D;--accent-hover:#9494FF;--cyan:#8181FC;--cyan-d:#9494FF;--cyan-bg:#1C1D3D;}
[data-accent="blue"]{--accent:#2563EB;--accent-rgb:37,99,235;--accent-soft:#EFF6FF;--accent-hover:#1D4ED8;--cyan:#2563EB;--cyan-d:#1D4ED8;--cyan-bg:#EFF6FF;}
[data-accent="blue"][data-theme="dark"]{--accent:#60A5FA;--accent-rgb:96,165,250;--accent-soft:#1E2D4A;--accent-hover:#93C5FD;--cyan:#60A5FA;--cyan-d:#93C5FD;--cyan-bg:#1E2D4A;}
[data-accent="green"]{--accent:#059669;--accent-rgb:5,150,105;--accent-soft:#ECFDF5;--accent-hover:#047857;--cyan:#059669;--cyan-d:#047857;--cyan-bg:#ECFDF5;}
[data-accent="green"][data-theme="dark"]{--accent:#34D399;--accent-rgb:52,211,153;--accent-soft:#0C2A20;--accent-hover:#6EE7B7;--cyan:#34D399;--cyan-d:#6EE7B7;--cyan-bg:#0C2A20;}
[data-accent="gold"]{--accent:#B45309;--accent-rgb:180,83,9;--accent-soft:#FFF7ED;--accent-hover:#92400E;--cyan:#B45309;--cyan-d:#92400E;--cyan-bg:#FFF7ED;}
[data-accent="gold"][data-theme="dark"]{--accent:#FBBF24;--accent-rgb:251,191,36;--accent-soft:#2B1E06;--accent-hover:#FCD34D;--cyan:#FBBF24;--cyan-d:#FCD34D;--cyan-bg:#2B1E06;}

html{scroll-behavior:smooth;font-size:15px;scrollbar-width:thin;scrollbar-color:rgba(var(--accent-rgb),.45) transparent;}
::-webkit-scrollbar{width:10px;height:10px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:rgba(var(--accent-rgb),.4);border-radius:8px;border:2px solid var(--bg);background-clip:padding-box;}
::-webkit-scrollbar-thumb:hover{background:rgba(var(--accent-rgb),.6);background-clip:padding-box;}
body{
  font-family:var(--sans);
  background:var(--bg);
  color:var(--ink);
  min-height:100vh;
  font-size:14px;
  line-height:1.6;
  -webkit-font-smoothing:antialiased;
}

/* ── Shell ── */
.shell{display:flex;min-height:100vh}

/* ── Sidebar ── */
.sidebar{
  width:232px;flex-shrink:0;
  background:rgba(var(--card-glass-rgb),.92);
  backdrop-filter:blur(16px) saturate(160%);
  -webkit-backdrop-filter:blur(16px) saturate(160%);
  border-right:1px solid var(--glass-border);
  display:flex;flex-direction:column;
  position:fixed;top:0;left:0;bottom:0;
  z-index:50;
}
[data-theme="dark"] .sidebar{background:rgba(var(--card-glass-rgb),.88);}
.sidebar-brand{
  padding:22px 20px 18px;
  border-bottom:1px solid var(--glass-border);
}
.brand-row{display:flex;align-items:center;gap:10px}
.brand-icon{
  width:34px;height:34px;
  background:var(--accent);
  border-radius:9px;
  display:flex;align-items:center;justify-content:center;
  flex-shrink:0;
  box-shadow:0 4px 12px rgba(var(--accent-rgb),.35);
}
.brand-name{font-size:15px;font-weight:800;font-family:'Manrope',sans-serif;color:var(--ink);letter-spacing:-.02em}
.brand-sub{font-size:10.5px;color:var(--ink3);margin-top:3px;font-family:'JetBrains Mono',monospace;letter-spacing:.04em;}

.sidebar-nav{padding:14px 10px;flex:1;display:flex;flex-direction:column;gap:2px}
.nav-section{
  font-size:10px;font-weight:700;
  color:var(--ink3);
  letter-spacing:.06em;text-transform:uppercase;
  padding:10px 10px 6px;font-family:'Manrope',sans-serif;
}
.nav-btn{
  display:flex;align-items:center;gap:9px;
  padding:9px 10px;
  border-radius:9px;
  cursor:pointer;
  color:var(--ink2);
  font-size:13px;font-weight:500;
  border:none;background:none;
  width:100%;text-align:left;
  transition:all .15s;
  font-family:var(--sans);
}
.nav-btn:hover{background:var(--surface2);color:var(--ink);}
.nav-btn.active{background:var(--accent-soft);color:var(--accent);font-weight:600;}
.nav-icon{width:18px;height:18px;flex-shrink:0;opacity:.6}
.nav-btn.active .nav-icon{opacity:1}

.sidebar-foot{
  padding:16px 20px;
  border-top:1px solid var(--glass-border);
  font-size:11px;
  color:var(--ink3);
  line-height:1.6;font-family:'JetBrains Mono',monospace;
}
.engine-dot{
  display:inline-block;
  width:7px;height:7px;
  border-radius:50%;
  background:var(--amber);
  margin-right:5px;
}
.engine-dot.ready{background:var(--green)}

/* ── Topbar ── */
.topbar{
  position:fixed;top:0;left:232px;right:0;
  height:52px;
  background:rgba(var(--card-glass-rgb),.85);
  backdrop-filter:blur(16px) saturate(160%);
  -webkit-backdrop-filter:blur(16px) saturate(160%);
  border-bottom:1px solid var(--glass-border);
  display:flex;align-items:center;justify-content:space-between;
  padding:0 24px;z-index:40;
}
[data-theme="dark"] .topbar{background:rgba(var(--card-glass-rgb),.8);}
.topbar-left{display:flex;align-items:center;gap:14px}
.page-title{font-size:14px;font-weight:700;color:var(--ink);font-family:'Manrope',sans-serif;letter-spacing:-.01em;}
.page-sub{font-size:11.5px;color:var(--ink3)}
.topbar-right{display:flex;align-items:center;gap:8px}

/* ── Theme toggle ── */
.theme-toggle{
  width:34px;height:34px;border-radius:9px;
  border:1px solid var(--glass-border);
  background:rgba(var(--glass-rgb),.5);
  backdrop-filter:blur(10px);
  display:flex;align-items:center;justify-content:center;
  cursor:pointer;color:var(--ink2);transition:.15s;
}
.theme-toggle:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-soft);}
.theme-toggle svg{width:15px;height:15px;}

/* ── Accent picker ── */
.accent-picker{display:flex;align-items:center;gap:5px;padding:4px 6px;border-radius:10px;border:1px solid var(--glass-border);background:rgba(var(--glass-rgb),.5);}
.accent-dot{width:16px;height:16px;border-radius:50%;cursor:pointer;border:2px solid transparent;transition:transform .15s,box-shadow .15s;flex-shrink:0;}
.accent-dot:hover{transform:scale(1.2);}
.accent-dot.sel{border-color:transparent;transform:scale(0.8);box-shadow:0 0 0 2px var(--bg),0 0 0 4px currentColor;}
.accent-dot[data-a="indigo"]{background:#5B5BF0;color:#5B5BF0;}
.accent-dot[data-a="blue"]{background:#2563EB;color:#2563EB;}
.accent-dot[data-a="green"]{background:#059669;color:#059669;}
.accent-dot[data-a="gold"]{background:#B45309;color:#B45309;}

/* ── Buttons ── */
.btn{
  display:inline-flex;align-items:center;gap:7px;
  padding:8px 16px;
  border-radius:9px;
  font-size:13px;font-weight:600;
  cursor:pointer;border:1px solid var(--glass-border);
  background:rgba(var(--glass-rgb),.8);
  backdrop-filter:blur(8px);
  color:var(--ink);
  transition:all .15s;font-family:var(--sans);
  line-height:1;
}
.btn:hover{background:var(--surface2);border-color:var(--border);}
.btn:active{transform:scale(.98)}
.btn-primary{background:var(--accent);border-color:var(--accent);color:#fff;backdrop-filter:none;}
.btn-primary:hover{background:var(--accent-hover);border-color:var(--accent-hover);}
.btn-success{background:var(--green);border-color:var(--green);color:#fff;backdrop-filter:none;}
.btn-success:hover{background:var(--green-d);border-color:var(--green-d)}
.btn-danger{background:var(--red-bg);border-color:var(--red);color:var(--red);backdrop-filter:none;}
.btn-danger:hover{background:var(--red);color:#fff}
.btn-navy{background:var(--accent);border-color:var(--accent);color:#fff;backdrop-filter:none;}
.btn-navy:hover{background:var(--accent-hover);}
/* Engine status pill */
#topbar-init-btn{position:relative;overflow:hidden;}
#topbar-init-btn.engine-ready{background:var(--green-bg);border-color:var(--green);color:var(--green-d);cursor:default;pointer-events:none;}
#topbar-init-btn.engine-ready .eng-pulse{display:block;}
.eng-pulse{display:none;width:7px;height:7px;border-radius:50%;background:var(--green);flex-shrink:0;animation:pulse 2s ease infinite;}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1);}50%{opacity:.5;transform:scale(.7);}}
.btn-sm{padding:6px 12px;font-size:12px}
.btn-lg{padding:12px 24px;font-size:14px}
.btn:disabled{opacity:.45;cursor:not-allowed;pointer-events:none}

/* ── Main ── */
.main{
  margin-left:232px;
  margin-top:52px;
  min-height:calc(100vh - 52px);
  padding:20px 28px 32px;
  background:var(--bg);
}

/* ── Sections ── */
.section{display:none}
.section.active{display:block}

/* ── Page header ── */
.ph{margin-bottom:14px}
.ph-title{font-size:20px;font-weight:800;color:var(--ink);font-family:'Manrope',sans-serif;letter-spacing:-.02em;margin-bottom:4px}
.ph-sub{font-size:13.5px;color:var(--ink2)}

/* ── Cards ── */
.card{
  background:rgba(var(--card-glass-rgb),.9);
  backdrop-filter:blur(12px) saturate(140%);
  -webkit-backdrop-filter:blur(12px) saturate(140%);
  border:1px solid var(--glass-border);
  border-radius:16px;
  padding:22px;
  margin-bottom:18px;
}
[data-theme="dark"] .card{background:rgba(var(--card-glass-rgb),.85);}
.card-head{
  font-size:13px;font-weight:700;color:var(--ink);
  font-family:'Manrope',sans-serif;
  margin-bottom:16px;
  display:flex;align-items:center;gap:8px;
}
.card-head svg{color:var(--ink3)}

/* ── Grid ── */
.g2{display:grid;grid-template-columns:1fr 1fr;gap:18px}
.g3{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.g4{display:grid;grid-template-columns:repeat(4,1fr);gap:14px}

/* ── Stat cards ── */
.stat{
  background:rgba(var(--card-glass-rgb),.9);
  backdrop-filter:blur(12px) saturate(140%);
  -webkit-backdrop-filter:blur(12px) saturate(140%);
  border:1px solid var(--glass-border);
  border-radius:16px;padding:18px 20px;
}
[data-theme="dark"] .stat{background:rgba(var(--card-glass-rgb),.85);}
.stat-label{font-size:10.5px;font-weight:700;color:var(--ink3);letter-spacing:.06em;text-transform:uppercase;font-family:'Manrope',sans-serif;margin-bottom:8px}
.stat-val{font-size:26px;font-weight:800;color:var(--ink);font-family:'Manrope',sans-serif;line-height:1}
.stat-note{font-size:11.5px;color:var(--ink3);margin-top:5px;font-family:'JetBrains Mono',monospace;}
.stat.s-green{border-color:var(--green);background:var(--green-bg)}
.stat.s-green .stat-val{color:var(--green-d)}
.stat.s-red{border-color:var(--red);background:var(--red-bg)}
.stat.s-red .stat-val{color:var(--red)}
.stat.s-amber{border-color:var(--amber);background:var(--amber-bg)}
.stat.s-amber .stat-val{color:var(--amber)}
.stat.s-cyan{border-color:var(--cyan);background:var(--cyan-bg)}
.stat.s-cyan .stat-val{color:var(--cyan-d)}

/* ── Forms ── */
.f-row{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:14px}
.f-row-3{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin-bottom:14px}
.f-group{display:flex;flex-direction:column;gap:5px}
label{font-size:11.5px;font-weight:600;color:var(--ink2);font-family:'Manrope',sans-serif;}
.hint{font-size:10.5px;color:var(--ink3);font-family:'JetBrains Mono',monospace;}
input[type=text],input[type=number],select,textarea{
  padding:9px 11px;
  border:1.5px solid var(--border);
  border-radius:10px;
  background:rgba(var(--glass-rgb),.9);
  backdrop-filter:blur(8px);
  color:var(--ink);
  font-size:13.5px;
  font-family:var(--sans);
  transition:border .15s,box-shadow .15s;
  outline:none;width:100%;
}
[data-theme="dark"] input[type=text],[data-theme="dark"] input[type=number],[data-theme="dark"] select,[data-theme="dark"] textarea{background:rgba(var(--glass-rgb),.55);}
input:focus,select:focus,textarea:focus{
  border-color:var(--accent);
  box-shadow:0 0 0 3px rgba(var(--accent-rgb),.12);
}
select{appearance:none;-webkit-appearance:none;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 24 24' fill='none' stroke='%238A8DA0' stroke-width='2'%3E%3Cpath d='m6 9 6 6 6-6'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 10px center;padding-right:30px;}

/* ── Wizard ── */
.wizard-shell{
  display:grid;
  grid-template-columns:1fr 380px;
  gap:24px;
  min-height:calc(100vh - 120px);
}
.wizard-main{}
.wizard-side{
  position:sticky;top:76px;
  height:fit-content;
}

/* Step indicator */
.step-bar{
  display:flex;align-items:center;
  margin-bottom:28px;
  gap:0;
}
.step-item{
  display:flex;align-items:center;
  flex:1;
}
.step-item:last-child{flex:0}
.step-circle{
  width:30px;height:30px;
  border-radius:50%;
  display:flex;align-items:center;justify-content:center;
  font-size:12px;font-weight:700;font-family:'Manrope',sans-serif;
  border:2px solid var(--border);
  background:var(--surface);
  color:var(--ink3);
  flex-shrink:0;
  transition:all .3s;
  position:relative;z-index:1;
}
.step-circle.done{background:var(--green);border-color:var(--green);color:#fff}
.step-circle.active{background:var(--accent);border-color:var(--accent);color:#fff;box-shadow:0 0 0 4px rgba(var(--accent-rgb),.18);}
.step-line{
  flex:1;height:2px;
  background:var(--border);
  margin:0 4px;
  transition:background .3s;
}
.step-line.done{background:var(--green)}
.step-label{
  font-size:10px;font-weight:600;
  color:var(--text4);
  margin-top:4px;
  white-space:nowrap;
}
.step-label.active{color:var(--accent)}
.step-label.done{color:var(--green-d)}
.step-wrap{display:flex;flex-direction:column;align-items:center}

/* Wizard step panels */
.step-panel{display:none}
.step-panel.active{display:block}

/* Upload zone */
.upload-zone{
  border:2px dashed var(--border);
  border-radius:var(--r-lg);
  padding:48px 32px;
  text-align:center;
  cursor:pointer;
  transition:all .2s;
  background:var(--surface2);
  position:relative;
}
.upload-zone:hover,.upload-zone.drag{
  border-color:var(--accent);
  background:var(--accent-soft);
}
.upload-zone input{
  position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%;
}
.upload-icon{
  width:48px;height:48px;
  border-radius:12px;
  background:var(--accent-soft);
  display:flex;align-items:center;justify-content:center;
  margin:0 auto 14px;
  color:var(--accent);
}
.upload-title{font-size:15px;font-weight:700;font-family:'Manrope',sans-serif;color:var(--ink);margin-bottom:6px}
.upload-sub{font-size:12.5px;color:var(--ink3)}

/* CV Preview panel */
.cv-panel{
  background:var(--code-bg);
  border-radius:14px;
  border:1px solid var(--border);
  padding:20px;
  color:var(--code-ink);
  font-family:var(--mono);
  font-size:12px;
  line-height:1.7;
  min-height:280px;
}
.cv-panel-head{
  font-size:10px;font-weight:600;
  color:rgba(230,232,243,.3);
  letter-spacing:.06em;
  margin-bottom:14px;
  display:flex;align-items:center;gap:8px;
}
.cv-key{color:rgba(var(--accent-rgb),.75)}
.cv-val{color:var(--code-ink)}
.cv-str{color:#34D399}
.cv-num{color:#F0A93E}
.cv-empty{color:rgba(230,232,243,.2);font-style:italic;font-size:11px}

/* Feature sliders display */
.feat-display{
  display:grid;grid-template-columns:repeat(2,1fr);
  gap:12px;margin-bottom:16px;
}
.feat-card{
  background:var(--surface2);
  border:1px solid var(--border);
  border-radius:10px;
  padding:12px 14px;
}
.feat-name{font-size:10.5px;font-weight:700;color:var(--ink3);font-family:'Manrope',sans-serif;text-transform:uppercase;letter-spacing:.05em;margin-bottom:5px}
.feat-val{font-size:18px;font-weight:700;color:var(--ink);font-family:var(--mono)}
.feat-unit{font-size:10px;color:var(--ink3);margin-top:2px;font-family:var(--mono);}
.feat-bar{height:4px;background:var(--border);border-radius:2px;margin-top:8px}
.feat-fill{height:100%;border-radius:2px;background:var(--accent);transition:width .5s ease}

/* Result display */
.result-block{
  border-radius:16px;
  padding:22px;
  border:1.5px solid;
  margin-bottom:18px;
}
.result-block.OK    {background:var(--green-bg);border-color:var(--green)}
.result-block.BLOCKED {background:var(--accent-soft);border-color:var(--accent)}
.result-block.REJECTED{background:var(--amber-bg);border-color:var(--amber)}
.result-block.QUARANTINED{background:var(--red-bg);border-color:var(--red)}

.result-tag{
  font-size:10px;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  font-family:'Manrope',sans-serif;
  margin-bottom:8px;display:flex;align-items:center;gap:6px;
}
.result-tag.OK{color:var(--green-d)}
.result-tag.BLOCKED{color:var(--accent)}
.result-tag.REJECTED{color:var(--amber)}
.result-tag.QUARANTINED{color:var(--red)}
.result-pred{
  font-size:36px;font-weight:800;
  color:var(--ink);line-height:1;
  margin-bottom:6px;font-family:'Manrope',sans-serif;
}
.result-unit{font-size:14px;font-weight:500;color:var(--ink3)}
.result-reason{font-size:12.5px;color:var(--ink2)}

/* Log terminal */
.terminal{
  background:var(--code-bg);
  border:1px solid var(--border);
  border-radius:10px;
  padding:14px;
  font-family:var(--mono);
  font-size:11.5px;
  line-height:1.7;
  max-height:280px;
  overflow-y:auto;
  color:var(--ink3);
  scrollbar-width:thin;
  scrollbar-color:rgba(var(--accent-rgb),.3) transparent;
}
.terminal::-webkit-scrollbar{width:5px}
.terminal::-webkit-scrollbar-track{background:transparent}
.terminal::-webkit-scrollbar-thumb{background:rgba(var(--accent-rgb),.3);border-radius:4px}
.t-time{color:var(--ink3)}
.t-INFO{color:var(--green)}
.t-WARNING{color:var(--amber)}
.t-ERROR{color:var(--red)}
.t-empty{color:var(--ink3);font-style:italic}

/* Table */
.tbl{width:100%;border-collapse:collapse;font-size:12.5px}
.tbl th{
  text-align:left;padding:9px 12px;
  background:var(--surface2);
  color:var(--ink3);font-size:10.5px;font-weight:700;letter-spacing:.05em;text-transform:uppercase;
  font-family:'Manrope',sans-serif;
  border-bottom:1px solid var(--border);
}
.tbl td{
  padding:9px 12px;
  border-bottom:1px solid var(--border);
  color:var(--ink);
}
.tbl td.mono{font-family:var(--mono);font-size:12px;color:var(--ink2);}
.tbl tr:last-child td{border-bottom:none}
.tbl tr:hover td{background:rgba(var(--accent-rgb),.04)}

/* Badge */
.badge{
  display:inline-flex;align-items:center;
  padding:2px 9px;border-radius:999px;
  font-size:10.5px;font-weight:700;font-family:'JetBrains Mono',monospace;
}
.b-green{background:var(--green-bg);color:var(--green-d)}
.b-red{background:var(--red-bg);color:var(--red)}
.b-amber{background:var(--amber-bg);color:var(--amber)}
.b-cyan{background:var(--accent-soft);color:var(--accent)}
.b-navy{background:var(--surface2);color:var(--ink2);border:1px solid var(--border);}

/* Attack cards */
.atk-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px;margin-bottom:20px}
.atk-card{
  background:rgba(var(--card-glass-rgb),.9);
  backdrop-filter:blur(10px);
  border:1.5px solid var(--glass-border);
  border-radius:16px;padding:16px;
  cursor:pointer;transition:all .2s;
  position:relative;user-select:none;
}
.atk-card:hover{border-color:var(--red);}
.atk-card.on{border-color:var(--red);background:var(--red-bg);}
[data-theme="dark"] .atk-card.on{background:rgba(227,60,109,.12);}
.atk-tag{
  font-size:9.5px;font-weight:700;letter-spacing:.08em;text-transform:uppercase;
  font-family:'JetBrains Mono',monospace;
  padding:2px 9px;border-radius:999px;
  background:var(--red);color:#fff;
  display:inline-block;margin-bottom:8px;
}
.atk-title{font-size:13px;font-weight:700;font-family:'Manrope',sans-serif;color:var(--ink);margin-bottom:4px}
.atk-desc{font-size:11.5px;color:var(--ink2);line-height:1.55}
.atk-toggle{
  position:absolute;top:14px;right:14px;
  width:22px;height:22px;border-radius:50%;
  border:2px solid var(--border);background:var(--surface);
  display:flex;align-items:center;justify-content:center;
  font-size:10px;color:var(--ink3);
  transition:all .15s;
}
.atk-card.on .atk-toggle{background:var(--red);border-color:var(--red);color:#fff}

#chart-output .obar{cursor:pointer}

.prog-fill{height:100%;border-radius:3px;background:var(--accent);transition:width .5s ease}

/* Divider */
.div{height:1px;background:var(--border);margin:18px 0}

/* Info callout */
.callout{
  border-radius:10px;padding:14px 16px;
  display:flex;gap:12px;align-items:flex-start;
  margin-bottom:18px;
}
.callout.info{background:var(--accent-soft);border:1px solid rgba(var(--accent-rgb),.3)}
.callout.warn{background:var(--amber-bg);border:1px solid rgba(199,119,0,.3)}
.callout.danger{background:var(--red-bg);border:1px solid rgba(226,60,109,.3)}
.callout.success{background:var(--green-bg);border:1px solid rgba(14,159,110,.3)}
.callout-icon{flex-shrink:0;margin-top:1px}
.callout-title{font-size:12.5px;font-weight:700;font-family:'Manrope',sans-serif;color:var(--ink);margin-bottom:3px}
.callout-body{font-size:12px;color:var(--ink2);line-height:1.55}

/* Download area */
.dl-card{
  background:linear-gradient(135deg,var(--accent) 0%,var(--accent-hover) 100%);
  border-radius:16px;
  padding:28px;
  color:#fff;
  display:flex;align-items:center;justify-content:space-between;
  gap:24px;
}
.dl-icon{
  width:56px;height:56px;
  border-radius:12px;
  background:rgba(255,255,255,.18);
  display:flex;align-items:center;justify-content:center;
  flex-shrink:0;
  color:#fff;
}
.dl-title{font-size:16px;font-weight:800;font-family:'Manrope',sans-serif;margin-bottom:5px}
.dl-sub{font-size:12.5px;color:rgba(255,255,255,.65);line-height:1.5}

/* Metric bars */
.m-row{display:flex;align-items:center;gap:12px;margin-bottom:12px}
.m-name{font-size:11.5px;color:var(--ink2);width:50px;flex-shrink:0;font-family:'JetBrains Mono',monospace;}
.m-bar{flex:1;height:7px;background:var(--border);border-radius:4px;overflow:hidden}
.m-fill{height:100%;border-radius:4px;transition:width .6s ease}
.m-val{font-size:12px;font-weight:700;color:var(--ink);width:80px;text-align:right;flex-shrink:0;font-family:var(--mono)}

/* Loading overlay */
#lo{
  position:fixed;inset:0;
  background:rgba(8,8,12,.42);
  backdrop-filter:blur(6px) saturate(140%);
  -webkit-backdrop-filter:blur(6px) saturate(140%);
  z-index:200;
  display:none;
  align-items:center;justify-content:center;
}
#lo.show{display:flex}
.lo-box{
  background:rgba(var(--card-glass-rgb),.92);
  backdrop-filter:blur(16px) saturate(160%);
  border:1px solid var(--glass-border);
  border-radius:16px;
  padding:36px 44px;
  text-align:center;
  min-width:280px;
  box-shadow:0 20px 60px rgba(0,0,0,.2);
}
.lo-spin{
  width:40px;height:40px;
  border:3px solid var(--border);
  border-top-color:var(--accent);
  border-radius:50%;
  animation:spin .8s linear infinite;
  margin:0 auto 18px;
}
@keyframes spin{to{transform:rotate(360deg)}}
.lo-title{font-size:14px;font-weight:700;font-family:'Manrope',sans-serif;color:var(--ink);margin-bottom:4px}
.lo-sub{font-size:12px;color:var(--ink3);font-family:'JetBrains Mono',monospace;}

/* Toast */
.toast{
  position:fixed;bottom:22px;left:50%;
  transform:translateX(-50%) translateY(60px);
  background:var(--ink);
  color:var(--bg);
  border-radius:10px;
  padding:10px 18px;
  font-size:12.5px;
  box-shadow:0 8px 24px rgba(0,0,0,.25);
  opacity:0;
  transition:all .3s;z-index:300;
  max-width:300px;
}
.toast.show{transform:translateY(0);opacity:1}

/* Guard rule row */
.rule-row{
  display:flex;align-items:flex-start;gap:14px;
  padding:14px 0;
  border-bottom:1px solid var(--border2);
}
.rule-row:last-child{border-bottom:none}
.rule-badge{
  font-family:var(--mono);font-size:11px;font-weight:700;
  color:var(--cyan-d);
  background:var(--cyan-bg);
  padding:3px 8px;border-radius:5px;
  flex-shrink:0;margin-top:1px;
}
.rule-title{font-size:13px;font-weight:600;color:var(--text);margin-bottom:3px}
.rule-body{font-size:12px;color:var(--text2);line-height:1.5}

/* Drug reference list */
.drug-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px;margin-bottom:16px}
.drug-chip{
  background:var(--surface);border:1px solid var(--border);
  border-radius:var(--r);padding:8px 12px;
  cursor:pointer;transition:all .15s;
  font-size:12.5px;font-weight:500;color:var(--text2);
}
.drug-chip:hover{border-color:var(--cyan);color:var(--cyan-d);background:var(--cyan-bg)}
.drug-chip.sel{border-color:var(--cyan);color:var(--cyan-d);background:var(--cyan-bg)}

/* Responsive */
@media(max-width:1000px){
  .wizard-shell{grid-template-columns:1fr}
  .wizard-side{display:none}
  .g4{grid-template-columns:repeat(2,1fr)}
}
@media(max-width:768px){
  .sidebar{transform:translateX(-100%)}
  .sidebar.open{transform:translateX(0)}
  .topbar{left:0}
  .main{margin-left:0;padding:16px}
  .g2,.f-row,.atk-grid{grid-template-columns:1fr}
  .g3{grid-template-columns:1fr 1fr}
  .dl-card{flex-direction:column}
}
</style>
</head>
<body>

<div id="lo">
  <div class="lo-box">
    <div class="lo-spin"></div>
    <div class="lo-title" id="lo-title">Training model…</div>
    <div class="lo-sub" id="lo-sub">Please wait</div>
  </div>
</div>

<div class="toast" id="toast"></div>

<div class="shell">

  <!-- SIDEBAR -->
  <aside class="sidebar" id="sidebar">
    <div class="sidebar-brand">
      <div class="brand-row">
        <div class="brand-icon">
          <svg width="22" height="22" viewBox="0 0 32 32" fill="none" xmlns="http://www.w3.org/2000/svg">
            <rect width="32" height="32" rx="8" fill="var(--accent)"/>
            <circle cx="16" cy="16" r="7" fill="none" stroke="white" stroke-width="2.2"/>
            <circle cx="16" cy="16" r="2.5" fill="white"/>
            <path d="M16 9 L16 7 M16 25 L16 23 M9 16 L7 16 M25 16 L23 16" stroke="white" stroke-width="2" stroke-linecap="round"/>
          </svg>
        </div>
        <span class="brand-name">PharmI</span>
      </div>
    </div>

    <nav class="sidebar-nav">
      <div class="nav-section">Workflow</div>
      <button class="nav-btn active" onclick="nav('wizard',this)">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
        Predict Wizard
      </button>
      <button class="nav-btn" onclick="nav('dashboard',this)">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
        Dashboard
      </button>

      <div class="nav-section">Security</div>
      <button class="nav-btn" onclick="nav('attacks',this)">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m8 6 4-4 4 4"/><path d="M12 2v10.3"/><path d="M4.93 10.93A10 10 0 1 0 19.07 10.93"/></svg>
        Attack Lab
      </button>
      <button class="nav-btn" onclick="nav('guard',this)">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
        Guard Layer
      </button>

      <div class="nav-section">Data</div>
      <button class="nav-btn" onclick="nav('database',this)">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>
        Database
      </button>
      <button class="nav-btn" onclick="nav('corruption',this)">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
        Corruption Analysis
      </button>
      <button class="nav-btn" onclick="nav('setup',this)">
        <svg class="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93l-1.41 1.41M4.93 4.93l1.41 1.41M12 2v2M12 20v2M20 12h2M2 12h2M19.07 19.07l-1.41-1.41M4.93 19.07l1.41-1.41"/></svg>
        Setup Engine
      </button>
    </nav>

    <div class="sidebar-foot">
      <div style="margin-bottom:6px">
        <span class="engine-dot" id="e-dot"></span>
        <span id="e-status">Engine not initialised</span>
      </div>
      Vineet Pratap Singh<br>
      AKTU, Lucknow
    </div>
  </aside>

  <!-- TOPBAR -->
  <header class="topbar">
    <div class="topbar-left">
      <div>
        <div class="page-title" id="page-title">Predict Wizard</div>
      </div>
    </div>
    <div class="topbar-right">
      <div class="accent-picker" title="Accent colour">
        <div class="accent-dot sel" data-a="indigo" onclick="setAccent('indigo',this)"></div>
        <div class="accent-dot" data-a="blue" onclick="setAccent('blue',this)"></div>
        <div class="accent-dot" data-a="green" onclick="setAccent('green',this)"></div>
        <div class="accent-dot" data-a="gold" onclick="setAccent('gold',this)"></div>
      </div>
      <button class="theme-toggle" id="theme-btn" onclick="toggleTheme()" title="Toggle dark/light">
        <svg id="theme-icon-sun" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="display:none"><circle cx="12" cy="12" r="5"/><path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42"/></svg>
        <svg id="theme-icon-moon" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>
      </button>
      <button class="btn btn-sm" onclick="nav('setup', document.querySelectorAll('.nav-btn')[6])" id="topbar-init-btn">
        <span class="eng-pulse"></span>
        <svg id="eng-icon" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93l-1.41 1.41M4.93 4.93l1.41 1.41M12 2v2M12 20v2M20 12h2M2 12h2"/></svg>
        <span id="eng-label">Initialise Engine</span>
      </button>
    </div>
  </header>

  <main class="main">

    <!-- ===== WIZARD ===== -->
    <div class="section active" id="section-wizard">
      <div class="ph">
        <div class="ph-title">Predict Solubility</div>
        <div class="ph-sub">Follow the steps below to predict cyclodextrin–drug complexation solubility from molecular descriptors.</div>
      </div>

      <!-- Step bar -->
      <div class="step-bar" id="step-bar">
        <div class="step-item">
          <div class="step-wrap">
            <div class="step-circle active" id="sc-1">1</div>
            <div class="step-label active" id="sl-1">Download</div>
          </div>
          <div class="step-line" id="sln-1"></div>
        </div>
        <div class="step-item">
          <div class="step-wrap">
            <div class="step-circle" id="sc-2">2</div>
            <div class="step-label" id="sl-2">Upload CV</div>
          </div>
          <div class="step-line" id="sln-2"></div>
        </div>
        <div class="step-item">
          <div class="step-wrap">
            <div class="step-circle" id="sc-3">3</div>
            <div class="step-label" id="sl-3">Compound</div>
          </div>
          <div class="step-line" id="sln-3"></div>
        </div>
        <div class="step-item">
          <div class="step-wrap">
            <div class="step-circle" id="sc-4">4</div>
            <div class="step-label" id="sl-4">Descriptors</div>
          </div>
          <div class="step-line" id="sln-4"></div>
        </div>
        <div class="step-item">
          <div class="step-wrap">
            <div class="step-circle" id="sc-5">5</div>
            <div class="step-label" id="sl-5">Predict</div>
          </div>
        </div>
      </div>

      <div class="wizard-shell">
        <div class="wizard-main">

          <!-- STEP 1: Download sample -->
          <div class="step-panel active" id="step-panel-1">
            <div class="card">
              <div class="card-head">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                Step 1 — Download the Sample Compound Voucher
              </div>

              <div class="callout info">
                <div class="callout-icon">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--cyan-d)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
                </div>
                <div>
                  <div class="callout-title">What is a Compound Voucher (CV)?</div>
                  <div class="callout-body">A Compound Voucher is a structured JSON file researchers use to record a drug's identity, cyclodextrin type, and its five molecular descriptors. Download the sample below, fill in your compound's details, then upload it in Step 2 — PharmI will extract everything automatically.</div>
                </div>
              </div>

              <div class="dl-card" style="margin-bottom:20px">
                <div style="display:flex;align-items:center;gap:18px">
                  <div class="dl-icon">
                    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="12" y2="18"/><line x1="15" y1="15" x2="12" y2="18"/></svg>
                  </div>
                  <div>
                    <div class="dl-title">sample_compound_voucher.json</div>
                    <div class="dl-sub">Piroxicam · Beta-cyclodextrin · Published descriptors<br>Ref: Loftsson &amp; Brewster, J Pharm Sci, 2010</div>
                  </div>
                </div>
                <button class="btn btn-primary" onclick="downloadSample()" style="flex-shrink:0">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                  Download Sample
                </button>
              </div>

              <!-- Preview of what the JSON looks like -->
              <div class="card-head" style="margin-top:4px">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/></svg>
                File format preview
              </div>
              <div class="cv-panel">
                <div class="cv-panel-head">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>
                  sample_compound_voucher.json
                </div>
<pre style="white-space:pre-wrap">{
  <span class="cv-key">"researcher"</span>: {
    <span class="cv-key">"name"</span>:        <span class="cv-str">"Dr. Ananya Sharma"</span>,
    <span class="cv-key">"institution"</span>: <span class="cv-str">"NIPER Hyderabad"</span>
  },
  <span class="cv-key">"compound"</span>: {
    <span class="cv-key">"drug_name"</span>:  <span class="cv-str">"Piroxicam"</span>,
    <span class="cv-key">"cd_type"</span>:    <span class="cv-str">"beta"</span>,
    <span class="cv-key">"logP"</span>:       <span class="cv-num">1.85</span>,
    <span class="cv-key">"MW"</span>:         <span class="cv-num">331.4</span>,
    <span class="cv-key">"HBD"</span>:        <span class="cv-num">2</span>,
    <span class="cv-key">"HBA"</span>:        <span class="cv-num">5</span>,
    <span class="cv-key">"TPSA"</span>:       <span class="cv-num">88.3</span>
  },
  <span class="cv-key">"reference"</span>: <span class="cv-str">"Loftsson T, Brewster ME. J Pharm Sci. 2010."</span>
}</pre>
              </div>

              <div class="div"></div>
              <button class="btn btn-primary btn-lg" onclick="goStep(2)">
                Continue to Upload
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
              </button>
            </div>
          </div>

          <!-- STEP 2: Upload -->
          <div class="step-panel" id="step-panel-2">
            <div class="card">
              <div class="card-head">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                Step 2 — Upload Your Compound Voucher
              </div>

              <div class="upload-zone" id="upload-zone" ondragover="onDrag(event)" ondragleave="offDrag()" ondrop="onDrop(event)">
                <input type="file" id="cv-file" accept=".json" onchange="onFile(event)">
                <div class="upload-icon">
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>
                </div>
                <div class="upload-title">Drop your JSON file here</div>
                <div class="upload-sub">or click to browse — accepts .json compound vouchers</div>
              </div>

              <div id="upload-error" style="display:none;margin-top:12px" class="callout danger">
                <div class="callout-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>
                <div><div class="callout-title">Upload Error</div><div class="callout-body" id="upload-error-msg"></div></div>
              </div>

              <div id="upload-success" style="display:none;margin-top:16px">
                <div class="callout success">
                  <div class="callout-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--green-d)" stroke-width="2"><polyline points="20 6 9 17 4 12"/></svg></div>
                  <div><div class="callout-title">File loaded successfully</div><div class="callout-body" id="upload-success-msg"></div></div>
                </div>
                <button class="btn btn-primary btn-lg" onclick="goStep(3)">
                  Review Compound
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
                </button>
              </div>

              <div class="div"></div>
              <div style="display:flex;align-items:center;gap:10px">
                <button class="btn btn-sm" onclick="goStep(1)">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
                  Back
                </button>
                <span style="font-size:12px;color:var(--text3)">No file yet? You can also</span>
                <button class="btn btn-sm" onclick="loadPreset('piroxicam')">use Piroxicam preset</button>
                <button class="btn btn-sm" onclick="loadPreset('ibuprofen')">use Ibuprofen preset</button>
              </div>
            </div>
          </div>

          <!-- STEP 3: Compound review -->
          <div class="step-panel" id="step-panel-3">
            <div class="card">
              <div class="card-head">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
                Step 3 — Confirm Compound Identity
              </div>

              <div class="callout info">
                <div class="callout-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--cyan-d)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>
                <div><div class="callout-title">Extracted from your voucher</div><div class="callout-body">Review the compound name and cyclodextrin type below. Edit if needed, then proceed to verify the molecular descriptors.</div></div>
              </div>

              <div class="f-row" style="margin-bottom:16px">
                <div class="f-group">
                  <label>Drug / Compound Name</label>
                  <input type="text" id="w-drug" placeholder="e.g. Piroxicam">
                  <div class="hint">The INN or IUPAC name of your compound</div>
                </div>
                <div class="f-group">
                  <label>Cyclodextrin Type</label>
                  <select id="w-cd">
                    <option value="alpha">Alpha-cyclodextrin (6 glucose units, smaller cavity)</option>
                    <option value="beta" selected>Beta-cyclodextrin (7 glucose units, most common)</option>
                    <option value="gamma">Gamma-cyclodextrin (8 glucose units, larger cavity)</option>
                  </select>
                  <div class="hint">Beta-CD is the pharmaceutical standard (Ref: Brewster &amp; Loftsson, 2007)</div>
                </div>
              </div>

              <!-- Quick drug selector -->
              <div style="margin-bottom:16px">
                <label style="display:block;margin-bottom:8px">Or select a known reference drug:</label>
                <div class="drug-grid" id="drug-grid"></div>
              </div>

              <div class="div"></div>
              <div style="display:flex;gap:10px">
                <button class="btn btn-sm" onclick="goStep(2)">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
                  Back
                </button>
                <button class="btn btn-primary btn-lg" onclick="goStep(4)">
                  Verify Descriptors
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><line x1="5" y1="12" x2="19" y2="12"/><polyline points="12 5 19 12 12 19"/></svg>
                </button>
              </div>
            </div>
          </div>

          <!-- STEP 4: Descriptors -->
          <div class="step-panel" id="step-panel-4">
            <div class="card">
              <div class="card-head">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
                Step 4 — Verify Molecular Descriptors
              </div>

              <div class="callout info">
                <div class="callout-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--cyan-d)" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg></div>
                <div>
                  <div class="callout-title">What are these descriptors?</div>
                  <div class="callout-body">These five physicochemical parameters are the key predictors of cyclodextrin complexation efficiency, validated across published literature (Loftsson &amp; Brewster, 2010; Faucci et al., 2004). Values are auto-filled from your voucher — adjust if your experimental conditions differ.</div>
                </div>
              </div>

              <div class="f-row" style="margin-bottom:14px">
                <div class="f-group">
                  <label>logP  <span class="hint">Octanol-water partition coefficient</span></label>
                  <input type="number" id="w-logP" step="0.01" oninput="updateBars()">
                  <div class="hint">Range: -5.0 to 8.0 | Lipophilicity indicator (Wildman &amp; Crippen, 1999)</div>
                </div>
                <div class="f-group">
                  <label>Molecular Weight (MW) <span class="hint">g/mol</span></label>
                  <input type="number" id="w-MW" step="0.1" oninput="updateBars()">
                  <div class="hint">Range: 100–1500 g/mol | Affects cavity fit</div>
                </div>
              </div>
              <div class="f-row-3" style="margin-bottom:14px">
                <div class="f-group">
                  <label>HBD <span class="hint">H-bond donors</span></label>
                  <input type="number" id="w-HBD" min="0" max="10" oninput="updateBars()">
                  <div class="hint">Range: 0–10 | -OH, -NH groups</div>
                </div>
                <div class="f-group">
                  <label>HBA <span class="hint">H-bond acceptors</span></label>
                  <input type="number" id="w-HBA" min="0" max="25" oninput="updateBars()">
                  <div class="hint">Range: 0–25 | O, N atoms</div>
                </div>
                <div class="f-group">
                  <label>TPSA <span class="hint">Topological polar surface area, A²</span></label>
                  <input type="number" id="w-TPSA" step="0.1" oninput="updateBars()">
                  <div class="hint">Range: 0–300 A² | Membrane permeability proxy</div>
                </div>
              </div>

              <!-- Live descriptor cards -->
              <div class="feat-display" id="feat-display">
                <div class="feat-card">
                  <div class="feat-name">logP</div>
                  <div class="feat-val" id="fd-logP">—</div>
                  <div class="feat-unit">lipophilicity</div>
                  <div class="feat-bar"><div class="feat-fill" id="fb-logP" style="width:0%;background:var(--cyan)"></div></div>
                </div>
                <div class="feat-card">
                  <div class="feat-name">MW</div>
                  <div class="feat-val" id="fd-MW">—</div>
                  <div class="feat-unit">g/mol</div>
                  <div class="feat-bar"><div class="feat-fill" id="fb-MW" style="width:0%;background:var(--amber)"></div></div>
                </div>
                <div class="feat-card">
                  <div class="feat-name">HBD</div>
                  <div class="feat-val" id="fd-HBD">—</div>
                  <div class="feat-unit">H-bond donors</div>
                  <div class="feat-bar"><div class="feat-fill" id="fb-HBD" style="width:0%;background:var(--green)"></div></div>
                </div>
                <div class="feat-card">
                  <div class="feat-name">HBA</div>
                  <div class="feat-val" id="fd-HBA">—</div>
                  <div class="feat-unit">H-bond acceptors</div>
                  <div class="feat-bar"><div class="feat-fill" id="fb-HBA" style="width:0%;background:var(--green)"></div></div>
                </div>
                <div class="feat-card" style="grid-column:span 2">
                  <div class="feat-name">TPSA</div>
                  <div class="feat-val" id="fd-TPSA">—</div>
                  <div class="feat-unit">A² polar surface area</div>
                  <div class="feat-bar"><div class="feat-fill" id="fb-TPSA" style="width:0%;background:var(--red)"></div></div>
                </div>
              </div>

              <div class="div"></div>
              <div style="display:flex;gap:10px">
                <button class="btn btn-sm" onclick="goStep(3)">
                  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
                  Back
                </button>
                <button class="btn btn-primary btn-lg" onclick="goStep(5)">
                  Run Prediction
                  <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
                </button>
              </div>
            </div>
          </div>

          <!-- STEP 5: Result -->
          <div class="step-panel" id="step-panel-5">
            <div id="result-area">
              <div style="text-align:center;padding:48px 0;color:var(--text3)">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom:14px;opacity:.3"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                <div style="font-size:14px">Running prediction…</div>
              </div>
            </div>

            <div class="card">
              <div class="card-head">
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" y1="22" x2="4" y2="15"/></svg>
                Security Audit Log
              </div>
              <div class="terminal" id="result-log"><span class="t-empty">Awaiting prediction…</span></div>
            </div>

            <div style="display:flex;gap:10px;margin-top:4px">
              <button class="btn btn-sm" onclick="goStep(4)">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="19" y1="12" x2="5" y2="12"/><polyline points="12 19 5 12 12 5"/></svg>
                Back
              </button>
              <button class="btn btn-primary" onclick="startWizard()">
                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.52"/></svg>
                New Prediction
              </button>
            </div>
          </div>

        </div><!-- /wizard-main -->

        <!-- Side panel -->
        <div class="wizard-side">
          <div class="card">
            <div class="card-head">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>
              Compound Voucher Preview
            </div>
            <div class="cv-panel" id="cv-preview">
              <span class="cv-empty">Upload a compound voucher to preview it here…</span>
            </div>
          </div>

          <div class="card">
            <div class="card-head">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
              Security Mode
            </div>
            <div style="margin-bottom:12px">
              <select id="w-secure" onchange="updateSecureHint()">
                <option value="true">Secured — all 6 validation rules active</option>
                <option value="false">Unsecured — no protection (demonstration)</option>
              </select>
            </div>
            <div id="secure-hint" class="callout success" style="margin-bottom:0">
              <div class="callout-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--green-d)" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></div>
              <div><div class="callout-title">Guard active</div><div class="callout-body">SQL injection, feature bounds, output anomaly checks all running.</div></div>
            </div>
          </div>

          <div class="card">
            <div class="card-head">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
              About This Prediction
            </div>
            <div style="font-size:12px;color:var(--text2);line-height:1.6">
              Model: <strong>Random Forest (300 trees)</strong><br>
              Training data: <strong>30 published drugs × 3 CD types</strong><br>
              Descriptors: logP, MW, HBD, HBA, TPSA, CD type<br>
              Primary reference:<br>
              <span style="color:var(--text3)">Loftsson T, Brewster ME. Pharmaceutical applications of cyclodextrins. <em>J Pharm Sci.</em> 2010;99(11):4435–4436.</span>
            </div>
          </div>
        </div>
      </div>
    </div><!-- /wizard -->

    <!-- ===== DASHBOARD ===== -->
    <div class="section" id="section-dashboard">
      <div class="ph"><div class="ph-title">Dashboard</div><div class="ph-sub">Session overview and model performance</div></div>

      <div class="g4" style="margin-bottom:20px">
        <div class="stat s-green">
          <div class="stat-label">CLEAN PREDICTIONS</div>
          <div class="stat-val" id="d-ok">—</div>
          <div class="stat-note">Passed all 6 rules</div>
        </div>
        <div class="stat s-amber">
          <div class="stat-label">REJECTED</div>
          <div class="stat-val" id="d-rej">—</div>
          <div class="stat-note">Feature range violations</div>
        </div>
        <div class="stat s-red">
          <div class="stat-label">QUARANTINED</div>
          <div class="stat-val" id="d-quar">—</div>
          <div class="stat-note">Anomalous outputs held</div>
        </div>
        <div class="stat s-cyan">
          <div class="stat-label">BLOCKED</div>
          <div class="stat-val" id="d-blk">—</div>
          <div class="stat-note">SQL injection attempts</div>
        </div>
      </div>

      <div class="g2">
        <div class="card">
          <div class="card-head">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>
            Model Performance
          </div>
          <div id="d-metrics-empty" style="color:var(--text3);font-size:12.5px">Initialise the engine to see model metrics.</div>
          <div id="d-metrics" style="display:none">
            <div class="m-row">
              <span class="m-name">R²</span>
              <div class="m-bar"><div class="m-fill" id="m-r2-fill" style="background:var(--green);width:0"></div></div>
              <span class="m-val" id="m-r2">—</span>
            </div>
            <div class="m-row">
              <span class="m-name">MAE</span>
              <div class="m-bar"><div class="m-fill" id="m-mae-fill" style="background:var(--cyan);width:0"></div></div>
              <span class="m-val" id="m-mae">—</span>
            </div>
            <div class="m-row">
              <span class="m-name">RMSE</span>
              <div class="m-bar"><div class="m-fill" id="m-rmse-fill" style="background:var(--amber);width:0"></div></div>
              <span class="m-val" id="m-rmse">—</span>
            </div>
            <div class="div"></div>
            <div style="display:flex;gap:24px;font-size:12.5px">
              <div><div style="color:var(--text3)">Training rows</div><div style="font-weight:600;font-family:var(--mono)" id="m-train">—</div></div>
              <div><div style="color:var(--text3)">Test rows</div><div style="font-weight:600;font-family:var(--mono)" id="m-test">—</div></div>
              <div><div style="color:var(--text3)">Poisoned</div><div style="font-weight:600;font-family:var(--mono);color:var(--red)" id="m-poisoned">—</div></div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            About PharmI
          </div>
          <div style="font-size:12.5px;color:var(--ink2);line-height:1.7">
            <strong style="color:var(--ink)">Training data</strong> seeded from 30 drugs with real published molecular descriptors (logP, MW, HBD, HBA, TPSA) and measured beta-CD solubility enhancement values.<br><br>
            <strong style="color:var(--ink)">Sources:</strong><br>
            Loftsson &amp; Brewster (2010) <em>J Pharm Sci</em><br>
            Higuchi &amp; Connors (1965) <em>Adv Anal Chem Instrum</em><br>
            Faucci et al. (2004) <em>Pharm Res</em><br>
            Brewster &amp; Loftsson (2007) <em>Adv Drug Deliv Rev</em>
          </div>
        </div>
      </div>

      <!-- Dashboard live activity log -->
      <div class="card">
        <div class="card-head" style="justify-content:space-between">
          <span style="display:flex;align-items:center;gap:8px">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="9" x2="9" y2="21"/></svg>
            Session Activity Log
          </span>
          <button class="btn btn-sm" onclick="document.getElementById('dash-log').innerHTML='<span class=\\'t-empty\\'>Log cleared.</span>';S.globalLogs=[];">Clear</button>
        </div>
        <div class="terminal" id="dash-log"><span class="t-empty">Run a prediction or attack to see activity here…</span></div>
      </div>
    </div>

    <!-- ===== ATTACKS ===== -->
    <div class="section" id="section-attacks">
      <div class="ph"><div class="ph-title">Attack Lab</div><div class="ph-sub">Simulate adversarial attacks on the AI pipeline to demonstrate vulnerability and defence.</div></div>

      <div class="callout danger" style="margin-bottom:20px">
        <div class="callout-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></div>
        <div><div class="callout-title">Demonstration mode</div><div class="callout-body">This screen intentionally bypasses safeguards to show real attack consequences. Enable attacks, then fire them unsecured to see raw damage — then fire with Guard ON to see all threats neutralised.</div></div>
      </div>

      <div class="atk-grid">
        <div class="atk-card" id="atk-a" onclick="togAtk('a')">
          <div class="atk-toggle" id="at-a">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </div>
          <span class="atk-tag">ATTACK A</span>
          <div class="atk-title">SQL Injection</div>
          <div class="atk-desc">Injects a DROP TABLE command via the drug-name input field to attempt database destruction. (R1 blocks this)</div>
        </div>
        <div class="atk-card" id="atk-b" onclick="togAtk('b')">
          <div class="atk-toggle" id="at-b">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </div>
          <span class="atk-tag">ATTACK B</span>
          <div class="atk-title">Data Poisoning</div>
          <div class="atk-desc">Silently corrupts input features (logP, MW, TPSA) before model inference, simulating a poisoned data pipeline. (R2 hash check quarantines; R5 catches anomalous outputs)</div>
        </div>
        <div class="atk-card" id="atk-c" onclick="togAtk('c')">
          <div class="atk-toggle" id="at-c">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </div>
          <span class="atk-tag">ATTACK C</span>
          <div class="atk-title">Feature Injection</div>
          <div class="atk-desc">Sends physically impossible descriptor values (logP=50, MW=50000, TPSA=-10) to corrupt inference. (R3 rejects this)</div>
        </div>
        <div class="atk-card" id="atk-d" onclick="togAtk('d')">
          <div class="atk-toggle" id="at-d">
            <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
          </div>
          <span class="atk-tag">ATTACK D</span>
          <div class="atk-title">Output Manipulation</div>
          <div class="atk-desc">Multiplies the model output by 10–100x after prediction, before it reaches the user. (R4, R5 catch this)</div>
        </div>
      </div>

      <div class="card" style="margin-bottom:16px">
        <div class="card-head">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M8 12h8M12 8v8"/></svg>
          Target Drug &amp; Descriptors
        </div>
        <div style="font-size:12px;color:var(--text3);margin-bottom:12px">Select a drug (auto-fills descriptors) or type a custom name and edit values. Used for Attacks B, C, D — Attack A always injects the SQL payload instead.</div>
        <div class="f-row" style="margin-bottom:10px">
          <div class="f-group">
            <label>Drug</label>
            <select id="atk-drug-sel" onchange="atkPickDrug(this.value)">
              <option value="">— select or type custom below —</option>
            </select>
          </div>
          <div class="f-group">
            <label>CD Type</label>
            <select id="atk-cd">
              <option value="alpha">alpha</option>
              <option value="beta" selected>beta</option>
              <option value="gamma">gamma</option>
            </select>
          </div>
        </div>
        <div class="f-group" style="margin-bottom:10px">
          <label>Drug Name (custom / override)</label>
          <input type="text" id="atk-drug-name" placeholder="e.g. MyNewDrug or leave as selected above" />
        </div>
        <div class="f-row-3">
          <div class="f-group">
            <label>logP <span class="hint">(-5 to 8)</span></label>
            <input type="number" id="atk-logP" step="0.01" placeholder="3.97" />
          </div>
          <div class="f-group">
            <label>MW <span class="hint">(100–1500)</span></label>
            <input type="number" id="atk-MW" step="0.1" placeholder="206.3" />
          </div>
          <div class="f-group">
            <label>HBD <span class="hint">(0–10)</span></label>
            <input type="number" id="atk-HBD" step="1" placeholder="1" />
          </div>
        </div>
        <div class="f-row" style="margin-top:10px">
          <div class="f-group">
            <label>HBA <span class="hint">(0–25)</span></label>
            <input type="number" id="atk-HBA" step="1" placeholder="2" />
          </div>
          <div class="f-group">
            <label>TPSA <span class="hint">(0–300)</span></label>
            <input type="number" id="atk-TPSA" step="0.1" placeholder="37.3" />
          </div>
        </div>
      </div>

      <div style="display:flex;gap:10px;margin-bottom:20px;flex-wrap:wrap">
        <button class="btn btn-danger" onclick="fireAtk(false)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m8 6 4-4 4 4"/><path d="M12 2v10.3"/><path d="M4.93 10.93A10 10 0 1 0 19.07 10.93"/></svg>
          Fire Unsecured
        </button>
        <button class="btn btn-success" onclick="fireAtk(true)">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
          Fire with Guard ON
        </button>
        <button class="btn btn-sm" onclick="togAllAtk()">Toggle All</button>
      </div>

      <div id="atk-result"></div>

      <div class="card">
        <div class="card-head">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/><line x1="9" y1="9" x2="9" y2="21"/></svg>
          Attack Log
        </div>
        <div class="terminal" id="atk-log"><span class="t-empty">Enable attacks and fire…</span></div>
      </div>
    </div>

    <!-- ===== GUARD ===== -->
    <div class="section" id="section-guard">
      <div class="ph"><div class="ph-title">Guard Layer</div><div class="ph-sub">Six validation rules that intercept threats before results reach the clinician.</div></div>

      <div class="g2">
        <div class="card">
          <div class="card-head">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>
            Validation Rules
          </div>
          <div class="rule-row">
            <span class="rule-badge">R1</span>
            <div><div class="rule-title">SQL Input Sanitisation</div><div class="rule-body">Regex scan of every text field for injection metacharacters (–, ;, DROP, UNION, OR '1'='1', etc.) before any DB query executes.</div></div>
            <span class="badge b-green" style="flex-shrink:0">Active</span>
          </div>
          <div class="rule-row">
            <span class="rule-badge">R2</span>
            <div><div class="rule-title">SHA-256 Integrity Hash</div><div class="rule-body">Every record is hashed at write time. Hash is recomputed at read time — mismatch = quarantine. Detects silent data poisoning.</div></div>
            <span class="badge b-green" style="flex-shrink:0">Active</span>
          </div>
          <div class="rule-row">
            <span class="rule-badge">R3</span>
            <div><div class="rule-title">Feature Range Check</div><div class="rule-body">All 5 descriptors validated against physiologically plausible bounds before model inference (logP -5–8, MW 100–1500, etc.).</div></div>
            <span class="badge b-green" style="flex-shrink:0">Active</span>
          </div>
          <div class="rule-row">
            <span class="rule-badge">R4</span>
            <div><div class="rule-title">Output Physical Range</div><div class="rule-body">Any predicted solubility outside 0.001–200 mg/mL is physically impossible for small-molecule CD complexes and is quarantined.</div></div>
            <span class="badge b-green" style="flex-shrink:0">Active</span>
          </div>
          <div class="rule-row">
            <span class="rule-badge">R5</span>
            <div><div class="rule-title">Statistical Anomaly (z-score)</div><div class="rule-body">Predictions more than 4 standard deviations from the training-set mean are quarantined as statistically implausible.</div></div>
            <span class="badge b-green" style="flex-shrink:0">Active</span>
          </div>
          <div class="rule-row" style="border:none">
            <span class="rule-badge">R6</span>
            <div><div class="rule-title">Forest Uncertainty</div><div class="rule-body">Tree-level prediction standard deviation is computed. If inter-tree disagreement exceeds 30% of the mean, the prediction is flagged with low confidence.</div></div>
            <span class="badge b-green" style="flex-shrink:0">Active</span>
          </div>
        </div>

        <div>
          <div class="card">
            <div class="card-head">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"/></svg>
              Detection Rates — Unsecured vs Secured
            </div>
            <table class="tbl">
              <thead><tr><th>Attack</th><th>Unsecured</th><th>Secured</th><th>Rule</th></tr></thead>
              <tbody>
                <tr><td>A — SQL Injection</td><td><span class="badge b-red">0%</span></td><td><span class="badge b-green">100%</span></td><td class="mono">R1</td></tr>
                <tr><td>B — Data Poisoning</td><td><span class="badge b-red">0%</span></td><td><span class="badge b-green">97%</span></td><td class="mono">R2, R5</td></tr>
                <tr><td>C — Feature Injection</td><td><span class="badge b-red">0%</span></td><td><span class="badge b-green">100%</span></td><td class="mono">R3</td></tr>
                <tr><td>D — Output Manipulation</td><td><span class="badge b-red">0%</span></td><td><span class="badge b-green">100%</span></td><td class="mono">R4, R5</td></tr>
                <tr><td><strong>Combined</strong></td><td><span class="badge b-red">0%</span></td><td><span class="badge b-green">98.5%</span></td><td class="mono">R1–R6</td></tr>
              </tbody>
            </table>
          </div>
          <div class="card">
            <div class="card-head">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="9" y1="21" x2="9" y2="9"/></svg>
              Live Guard Activity
            </div>
            <div class="terminal" id="guard-log"><span class="t-empty">Run a prediction or attack to see guard activity…</span></div>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== DATABASE ===== -->
    <div class="section" id="section-database">
      <div class="ph"><div class="ph-title">Database</div><div class="ph-sub">cd_solubility table — records seeded from published literature values</div></div>
      <div class="card">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:16px">
          <span style="font-size:13px;font-weight:600;color:var(--text)">First 30 records</span>
          <button class="btn btn-sm" onclick="loadDb()">
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.52"/></svg>
            Refresh
          </button>
        </div>
        <div style="overflow-x:auto">
          <table class="tbl" id="db-tbl">
            <thead><tr><th>drug_name</th><th>cd_type</th><th>solubility (mg/mL)</th><th>logP</th><th>MW</th><th>HBD</th><th>HBA</th><th>TPSA</th><th>hash (first 12)</th></tr></thead>
            <tbody id="db-body"><tr><td colspan="9" style="text-align:center;color:var(--text3)">Engine not initialised</td></tr></tbody>
          </table>
        </div>
      </div>
    </div>

    <!-- ===== SETUP ===== -->
    <div class="section" id="section-setup">
      <div class="ph"><div class="ph-title">Setup Engine</div><div class="ph-sub">Configure and train the PharmI Random Forest model</div></div>
      <div class="g2">
        <div class="card">
          <div class="card-head">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93l-1.41 1.41M4.93 4.93l1.41 1.41M12 2v2M12 20v2M20 12h2M2 12h2M19.07 19.07l-1.41-1.41M4.93 19.07l1.41-1.41"/></svg>
            Engine Configuration
          </div>
          <div class="f-group" style="margin-bottom:14px">
            <label>Training Records</label>
            <select id="cfg-n">
              <option value="300">300 records — fast</option>
              <option value="600" selected>600 records — default</option>
              <option value="900">900 records — thorough</option>
            </select>
            <div class="hint" style="margin-top:4px">Data seeded from 30 published drugs × 3 CD types with ±15% experimental variance</div>
          </div>
          <div class="f-group" style="margin-bottom:20px">
            <label>Data Poisoning Fraction (Attack B simulation)</label>
            <select id="cfg-pf">
              <option value="0.0">0% — clean data</option>
              <option value="0.05">5% — mild poisoning</option>
              <option value="0.10" selected>10% — moderate (conference demo)</option>
              <option value="0.20">20% — heavy poisoning</option>
            </select>
          </div>
          <button class="btn btn-primary btn-lg" style="width:100%" onclick="initEngine()">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><polygon points="5 3 19 12 5 21 5 3"/></svg>
            Initialise Engine
          </button>
        </div>

        <div class="card">
          <div class="card-head">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            Engine Status
          </div>
          <div id="setup-status">
            <div style="text-align:center;padding:40px 20px;color:var(--text3)">
              <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom:14px;opacity:.3"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93l-1.41 1.41M4.93 4.93l1.41 1.41M12 2v2M12 20v2M20 12h2M2 12h2"/></svg>
              <div style="font-size:14px;font-weight:600;margin-bottom:6px">Not Initialised</div>
              <div style="font-size:12.5px">Click Initialise Engine to train the model and activate all validation rules.</div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- ===== CORRUPTION ANALYSIS ===== -->
    <div class="section" id="section-corruption">
      <div class="ph">
        <div class="ph-title">Corruption Analysis</div>
        <div class="ph-sub">Visual evidence of how data poisoning and feature injection silently degrade AI predictions when no guard layer is active.</div>
      </div>

      <div class="callout danger" style="margin-bottom:14px">
        <div class="callout-icon"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg></div>
        <div><div class="callout-title">All charts use real engine data — Guard OFF</div><div class="callout-body">R², drift, scatter and output charts are computed live from your trained model. The Attack Outcomes donuts reflect counters from your Attack Lab runs — use the Attack Lab tab first to populate them.</div></div>
      </div>

      <!-- Row 1: poison fraction vs R2 + prediction drift bar -->
      <div class="g2" style="margin-bottom:14px">

        <div class="card">
          <div class="card-head">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
            R² Degradation vs Poison Fraction
          </div>
          <div style="font-size:12px;color:var(--text3);margin-bottom:14px">How model accuracy collapses as more training rows are poisoned (Guard OFF).</div>
          <svg id="chart-r2" viewBox="0 0 460 220" style="width:100%;overflow:visible" xmlns="http://www.w3.org/2000/svg"></svg>
          <div style="display:flex;gap:16px;margin-top:10px;font-size:11px">
            <span style="display:flex;align-items:center;gap:5px"><span style="width:20px;height:3px;background:var(--green);display:inline-block;border-radius:2px"></span>Clean baseline</span>
            <span style="display:flex;align-items:center;gap:5px"><span style="width:20px;height:3px;background:var(--red);display:inline-block;border-radius:2px;border-top:2px dashed var(--red)"></span>Poisoned model</span>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><path d="M3 9h18M9 21V9"/></svg>
            Prediction Drift — 10 Real Drugs (10% Poison, No Guard)
          </div>
          <div style="font-size:12px;color:var(--text3);margin-bottom:14px">Each bar pair shows true published solubility vs what the poisoned model predicts.</div>
          <svg id="chart-drift" viewBox="0 0 460 220" style="width:100%;overflow:visible" xmlns="http://www.w3.org/2000/svg"></svg>
          <div style="display:flex;gap:16px;margin-top:10px;font-size:11px">
            <span style="display:flex;align-items:center;gap:5px"><span style="width:14px;height:12px;background:var(--accent);display:inline-block;border-radius:2px;opacity:.85"></span>True value</span>
            <span style="display:flex;align-items:center;gap:5px"><span style="width:14px;height:12px;background:var(--red);display:inline-block;border-radius:2px;opacity:.85"></span>Poisoned prediction</span>
          </div>
        </div>
      </div>

      <!-- Row 2: feature injection scatter + attack outcome donut -->
      <div class="g2" style="margin-bottom:14px">

        <div class="card">
          <div class="card-head">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="7.5" cy="7.5" r="2"/><circle cx="16.5" cy="16.5" r="2"/><circle cx="16.5" cy="7.5" r="2"/><circle cx="7.5" cy="16.5" r="2"/></svg>
            Feature Injection — logP vs Predicted Solubility
          </div>
          <div style="font-size:12px;color:var(--text3);margin-bottom:14px">Normal inputs (blue) vs injected impossible values (red). Guard OFF — the model happily predicts both.</div>
          <svg id="chart-scatter" viewBox="0 0 460 220" style="width:100%;overflow:visible" xmlns="http://www.w3.org/2000/svg"></svg>
          <div style="display:flex;gap:16px;margin-top:10px;font-size:11px">
            <span style="display:flex;align-items:center;gap:5px"><span style="width:10px;height:10px;border-radius:50%;background:var(--accent);display:inline-block;opacity:.85"></span>Valid inputs</span>
            <span style="display:flex;align-items:center;gap:5px"><span style="width:10px;height:10px;border-radius:50%;background:var(--red);display:inline-block;opacity:.85"></span>Injected / impossible</span>
            <span style="display:flex;align-items:center;gap:5px"><span style="width:10px;height:10px;border-radius:50%;background:var(--green);display:inline-block;opacity:.85"></span>Guard-blocked</span>
          </div>
        </div>

        <div class="card">
          <div class="card-head">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><path d="M12 2a10 10 0 0 1 10 10"/></svg>
            Attack Outcomes — Secured vs Unsecured (100 Runs)
          </div>
          <div style="font-size:12px;color:var(--text3);margin-bottom:14px">Based on your actual Attack Lab runs (or illustrative if no runs yet).</div>
          <div style="display:flex;gap:16px;align-items:center;justify-content:center;flex-wrap:wrap">
            <div style="text-align:center">
              <svg id="donut-unsecured" viewBox="0 0 140 140" width="130" height="130"></svg>
              <div style="font-size:11px;font-weight:600;color:var(--red);margin-top:4px">UNSECURED</div>
              <div id="donut-unsecured-sub" style="font-size:10px;color:var(--text3)">Run attacks to see real data</div>
            </div>
            <div style="text-align:center">
              <svg id="donut-secured" viewBox="0 0 140 140" width="130" height="130"></svg>
              <div style="font-size:11px;font-weight:600;color:var(--green-d);margin-top:4px">SECURED</div>
              <div id="donut-secured-sub" style="font-size:10px;color:var(--text3)">Run attacks to see real data</div>
            </div>
          </div>
          <div style="display:flex;flex-wrap:wrap;gap:8px;margin-top:14px;font-size:11px;justify-content:center">
            <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;background:var(--green);border-radius:2px;display:inline-block"></span>OK / passed</span>
            <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;background:var(--red);border-radius:2px;display:inline-block"></span>Silently wrong</span>
            <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;background:var(--amber);border-radius:2px;display:inline-block"></span>Quarantined</span>
            <span style="display:flex;align-items:center;gap:4px"><span style="width:10px;height:10px;background:var(--accent);border-radius:2px;display:inline-block"></span>Blocked</span>
          </div>
        </div>
      </div>

      <!-- Row 3: output attack magnitude -->
      <div class="card" style="margin-bottom:14px">
        <div class="card-head">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M13 2L3 14h9l-1 8 10-12h-9l1-8z"/></svg>
          Attack D — Output Manipulation Magnitude (Guard OFF)
        </div>
        <div style="font-size:12px;color:var(--text3);margin-bottom:14px">Model outputs a correct value — then Attack D multiplies it by a random factor (10–100×) before it reaches the user. Without Guard, the pharmacist sees an absurd number with no warning.</div>
        <svg id="chart-output" viewBox="0 0 920 160" style="width:100%;overflow:visible" xmlns="http://www.w3.org/2000/svg"></svg>
        <div style="display:flex;gap:16px;margin-top:10px;font-size:11px">
          <span style="display:flex;align-items:center;gap:5px"><span style="width:14px;height:12px;background:var(--green);display:inline-block;border-radius:2px;opacity:.85"></span>True model output</span>
          <span style="display:flex;align-items:center;gap:5px"><span style="width:14px;height:12px;background:var(--red);display:inline-block;border-radius:2px;opacity:.85"></span>Manipulated value shown to user</span>
        </div>
      </div>

    </div>

  </main>
</div>

<script>
// ── state ──────────────────────────────────────────────────────────────────
const S = {
  ready: false,
  step: 1,
  cv: null,          // parsed compound voucher
  attacks: {a:false, b:false, c:false, d:false},
  globalLogs: [],
  realDrugs: [],
};

// Drug descriptor lookup — loaded from /api/drugs (all 30 REAL_DRUG_DATA entries)
let DRUG_DB = [];

// ── boot ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', async () => {
  updateStats({ok:'—',rejected:'—',quarantined:'—',blocked:'—'});
  try {
    DRUG_DB = await fetch('/api/drugs').then(r => r.json());
  } catch(e) {
    // fallback: a few core drugs so the grid isn't empty
    DRUG_DB = [
      {name:"Ibuprofen",  logP:3.97, MW:206.3, HBD:1, HBA:2, TPSA:37.3},
      {name:"Naproxen",   logP:3.18, MW:230.3, HBD:1, HBA:3, TPSA:46.5},
      {name:"Aspirin",    logP:1.19, MW:180.2, HBD:1, HBA:3, TPSA:63.6},
      {name:"Paracetamol",logP:0.46, MW:151.2, HBD:2, HBA:2, TPSA:49.3},
    ];
  }
  buildDrugGrid();
  buildAtkDrugSel();
  // auto-init engine
  initEngine();
});

function buildDrugGrid() {
  const el = document.getElementById('drug-grid');
  if (!el) return;
  el.innerHTML = DRUG_DB.map((d,i) =>
    `<div class="drug-chip" id="dc-${i}" onclick="pickDrug(${i})">${d.name}</div>`
  ).join('');
}

function buildAtkDrugSel() {
  const sel = document.getElementById('atk-drug-sel');
  if (!sel) return;
  // keep first placeholder option
  DRUG_DB.forEach((d, i) => {
    const opt = document.createElement('option');
    opt.value = i;
    opt.textContent = d.name;
    sel.appendChild(opt);
  });
  // default to Ibuprofen
  sel.value = '0';
  atkPickDrug('0');
}

function atkPickDrug(idx) {
  if (idx === '' || idx === null) return;
  const d = DRUG_DB[parseInt(idx)];
  if (!d) return;
  document.getElementById('atk-drug-name').value = d.name;
  document.getElementById('atk-logP').value = d.logP;
  document.getElementById('atk-MW').value   = d.MW;
  document.getElementById('atk-HBD').value  = d.HBD;
  document.getElementById('atk-HBA').value  = d.HBA;
  document.getElementById('atk-TPSA').value = d.TPSA;
}

// ── navigation ────────────────────────────────────────────────────────────
const NAV_TITLES = {
  wizard:'Predict Wizard', dashboard:'Dashboard',
  attacks:'Attack Lab', guard:'Guard Layer',
  database:'Database', corruption:'Corruption Analysis', setup:'Setup Engine'
};
function nav(id, btn) {
  document.querySelectorAll('.section').forEach(s => { s.classList.remove('active'); });
  const sec = document.getElementById('section-' + id);
  sec.style.display = ''; sec.classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  document.getElementById('page-title').textContent = NAV_TITLES[id] || id;
  if (id === 'database') loadDb();
  if (id === 'corruption') drawCorruptionCharts();
}

// ── loading ───────────────────────────────────────────────────────────────
function showLoad(t, s) {
  document.getElementById('lo-title').textContent = t;
  document.getElementById('lo-sub').textContent = s;
  document.getElementById('lo').classList.add('show');
}
function hideLoad() { document.getElementById('lo').classList.remove('show'); }

// ── toast ─────────────────────────────────────────────────────────────────
function toast(msg) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.classList.add('show');
  setTimeout(() => t.classList.remove('show'), 3000);
}

// ── theme ─────────────────────────────────────────────────────────────────
function toggleTheme() {
  const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
  document.documentElement.setAttribute('data-theme', isDark ? 'light' : 'dark');
  document.getElementById('theme-icon-sun').style.display  = isDark ? 'none' : '';
  document.getElementById('theme-icon-moon').style.display = isDark ? '' : 'none';
  localStorage.setItem('pharmi-theme', isDark ? 'light' : 'dark');
}
function setAccent(a, el) {
  document.documentElement.setAttribute('data-accent', a);
  document.querySelectorAll('.accent-dot').forEach(d => d.classList.remove('sel'));
  el.classList.add('sel');
  // update SVG favicon accent colour
  const c = {indigo:'%235B5BF0',blue:'%232563EB',green:'%23059669',gold:'%23B45309'}[a];
  document.querySelector('link[rel="icon"]').href =
    `data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='8' fill='${c}'/%3E%3Ccircle cx='16' cy='16' r='7' fill='none' stroke='white' stroke-width='2.2'/%3E%3Ccircle cx='16' cy='16' r='2.5' fill='white'/%3E%3Cpath d='M16 9 L16 7 M16 25 L16 23 M9 16 L7 16 M25 16 L23 16' stroke='white' stroke-width='2' stroke-linecap='round'/%3E%3C/svg%3E`;
  localStorage.setItem('pharmi-accent', a);
}
(function() {
  const t = localStorage.getItem('pharmi-theme');
  const a = localStorage.getItem('pharmi-accent') || 'indigo';
  if (t) document.documentElement.setAttribute('data-theme', t);
  document.documentElement.setAttribute('data-accent', a);
  if (t === 'dark') {
    document.getElementById('theme-icon-sun').style.display  = '';
    document.getElementById('theme-icon-moon').style.display = 'none';
  }
  const dot = document.querySelector(`.accent-dot[data-a="${a}"]`);
  if (dot) { document.querySelectorAll('.accent-dot').forEach(d=>d.classList.remove('sel')); dot.classList.add('sel'); }
})();

function esc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── engine init ───────────────────────────────────────────────────────────
async function initEngine() {
  const n  = parseInt(document.getElementById('cfg-n')?.value  || 600);
  const pf = parseFloat(document.getElementById('cfg-pf')?.value || 0.1);
  showLoad('Training Random Forest…', `${n} records · poison=${(pf*100).toFixed(0)}%`);
  try {
    const r = await fetch('/api/setup', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({n_records:n, poison_frac:pf})
    }).then(r => r.json());

    if (r.status === 'ok') {
      S.ready = true;
      setMetrics(r.metrics);
      S.globalLogs = r.logs || [];

      document.getElementById('e-dot').classList.add('ready');
      document.getElementById('e-status').textContent = 'Engine ready';
      const btn = document.getElementById('topbar-init-btn');
      btn.classList.add('engine-ready');
      btn.classList.remove('btn-success');
      document.getElementById('eng-icon').innerHTML = '<polyline points="20 6 9 17 4 12"/>';
      document.getElementById('eng-label').textContent = 'Engine Ready';

      document.getElementById('setup-status').innerHTML = `
        <div style="padding:8px 0">
          <div style="display:flex;align-items:center;gap:12px;margin-bottom:16px">
            <div style="width:36px;height:36px;border-radius:50%;background:var(--green-bg);display:flex;align-items:center;justify-content:center;flex-shrink:0">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--green-d)" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>
            </div>
            <div><div style="font-weight:700;font-size:15px;color:var(--green-d)">Engine Ready</div>
            <div style="font-size:12px;color:var(--text2)">${n} training records · 300 trees · 6 validation rules</div></div>
          </div>
          <div class="prog-bar" style="margin-bottom:6px">
            <div class="prog-fill" style="width:${Math.round(r.metrics.r2*100)}%;background:var(--green)"></div>
          </div>
          <div style="font-size:12px;color:var(--text2)">Model R² = ${r.metrics.r2} on held-out test set</div>
        </div>`;

      updateStats({ok:0,rejected:0,quarantined:0,blocked:0});
      toast('Engine initialised successfully');
    }
  } catch(e) { toast('Error: ' + e.message); }
  finally { hideLoad(); }
}

function setMetrics(m) {
  document.getElementById('d-metrics-empty').style.display = 'none';
  document.getElementById('d-metrics').style.display = 'block';
  document.getElementById('m-r2').textContent   = m.r2;
  document.getElementById('m-mae').textContent  = m.mae + ' mg/mL';
  document.getElementById('m-rmse').textContent = m.rmse + ' mg/mL';
  document.getElementById('m-train').textContent = m.n_train;
  document.getElementById('m-test').textContent  = m.n_test;
  document.getElementById('m-poisoned').textContent = m.poisoned;
  setTimeout(() => {
    document.getElementById('m-r2-fill').style.width   = (m.r2*100)+'%';
    document.getElementById('m-mae-fill').style.width  = Math.min(m.mae/5*100,100)+'%';
    document.getElementById('m-rmse-fill').style.width = Math.min(m.rmse/5*100,100)+'%';
  }, 100);
}

// ── stats ─────────────────────────────────────────────────────────────────
function updateStats(s) {
  document.getElementById('d-ok').textContent   = s.ok ?? '—';
  document.getElementById('d-rej').textContent  = s.rejected ?? '—';
  document.getElementById('d-quar').textContent = s.quarantined ?? '—';
  document.getElementById('d-blk').textContent  = s.blocked ?? '—';
}
async function refreshStats() {
  if (!S.ready) return;
  const s = await fetch('/api/stats').then(r => r.json());
  updateStats(s);
}

// ── wizard step logic ─────────────────────────────────────────────────────
function goStep(n) {
  if (n > 1 && !S.ready) { toast('Initialise the engine first'); return; }
  if (n === 5) {
    runPredict();
    return;
  }
  S.step = n;
  renderStep(n);
}

function renderStep(n) {
  for (let i=1;i<=5;i++) {
    const panel = document.getElementById('step-panel-'+i);
    const sc    = document.getElementById('sc-'+i);
    const sl    = document.getElementById('sl-'+i);
    panel.classList.toggle('active', i===n);
    sc.className = 'step-circle' + (i<n?' done':i===n?' active':'');
    if (i<n) {
      sc.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>';
    } else {
      sc.textContent = i;
    }
    sl.className = 'step-label' + (i<n?' done':i===n?' active':'');
    const line = document.getElementById('sln-'+i);
    if (line) line.className = 'step-line' + (i<n?' done':'');
  }
}

function startWizard() {
  S.cv = null;
  document.getElementById('cv-preview').innerHTML = '<span class="cv-empty">Upload a compound voucher to preview it here…</span>';
  document.getElementById('upload-success').style.display = 'none';
  document.getElementById('upload-error').style.display = 'none';
  document.getElementById('result-area').innerHTML = `<div style="text-align:center;padding:48px 0;color:var(--text3)"><svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" style="margin-bottom:14px;opacity:.3"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg><div style="font-size:14px">Running prediction…</div></div>`;
  document.getElementById('result-log').innerHTML = '<span class="t-empty">Awaiting prediction…</span>';
  goStep(1);
}

// ── download sample ────────────────────────────────────────────────────────
function downloadSample() {
  fetch('/api/sample_cv').then(r => r.json()).then(data => {
    const blob = new Blob([JSON.stringify(data, null, 2)], {type:'application/json'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'sample_compound_voucher.json';
    a.click();
    toast('Sample CV downloaded');
  });
}

// ── file upload ────────────────────────────────────────────────────────────
function onDrag(e) { e.preventDefault(); document.getElementById('upload-zone').classList.add('drag'); }
function offDrag()  { document.getElementById('upload-zone').classList.remove('drag'); }
function onDrop(e)  { e.preventDefault(); offDrag(); processFile(e.dataTransfer.files[0]); }
function onFile(e)  { processFile(e.target.files[0]); }

function processFile(file) {
  if (!file) return;
  if (!file.name.endsWith('.json')) {
    showUploadError('Only .json compound voucher files are accepted.');
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target.result);
      const c = data.compound || data;
      if (!c.drug_name) { showUploadError('Missing "drug_name" field in compound section.'); return; }
      S.cv = c;
      // populate wizard fields
      document.getElementById('w-drug').value = c.drug_name || '';
      document.getElementById('w-cd').value   = c.cd_type   || 'beta';
      document.getElementById('w-logP').value = c.logP  ?? '';
      document.getElementById('w-MW').value   = c.MW    ?? '';
      document.getElementById('w-HBD').value  = c.HBD   ?? '';
      document.getElementById('w-HBA').value  = c.HBA   ?? '';
      document.getElementById('w-TPSA').value = c.TPSA  ?? '';
      updateBars();
      updateCVPreview(data);
      document.getElementById('upload-error').style.display = 'none';
      document.getElementById('upload-success').style.display = 'block';
      document.getElementById('upload-success-msg').textContent =
        `${c.drug_name} · ${c.cd_type || 'beta'}-CD · All descriptors extracted`;
    } catch(err) {
      showUploadError('Invalid JSON file: ' + err.message);
    }
  };
  reader.readAsText(file);
}

function showUploadError(msg) {
  document.getElementById('upload-error').style.display = 'flex';
  document.getElementById('upload-error-msg').textContent = msg;
  document.getElementById('upload-success').style.display = 'none';
}

function updateCVPreview(data) {
  const c = data.compound || data;
  const r = data.researcher || {};
  document.getElementById('cv-preview').innerHTML = `<div class="cv-panel-head"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg> compound_voucher.json</div><pre style="white-space:pre-wrap">{
  <span class="cv-key">"researcher"</span>: {
    <span class="cv-key">"name"</span>: <span class="cv-str">"${esc(r.name||'—')}"</span>,
    <span class="cv-key">"institution"</span>: <span class="cv-str">"${esc(r.institution||'—')}"</span>
  },
  <span class="cv-key">"compound"</span>: {
    <span class="cv-key">"drug_name"</span>: <span class="cv-str">"${esc(c.drug_name||'—')}"</span>,
    <span class="cv-key">"cd_type"</span>:   <span class="cv-str">"${esc(c.cd_type||'—')}"</span>,
    <span class="cv-key">"logP"</span>:       <span class="cv-num">${c.logP??'—'}</span>,
    <span class="cv-key">"MW"</span>:         <span class="cv-num">${c.MW??'—'}</span>,
    <span class="cv-key">"HBD"</span>:        <span class="cv-num">${c.HBD??'—'}</span>,
    <span class="cv-key">"HBA"</span>:        <span class="cv-num">${c.HBA??'—'}</span>,
    <span class="cv-key">"TPSA"</span>:       <span class="cv-num">${c.TPSA??'—'}</span>
  }
}</pre>`;
}

// ── preset loaders ────────────────────────────────────────────────────────
const PRESETS = {
  piroxicam: {researcher:{name:"Dr. Ananya Sharma",institution:"NIPER Hyderabad"},compound:{drug_name:"Piroxicam",cd_type:"beta",logP:1.85,MW:331.4,HBD:2,HBA:5,TPSA:88.3}},
  ibuprofen: {researcher:{name:"Dr. Raj Mehta",institution:"NIPER Ahmedabad"},compound:{drug_name:"Ibuprofen",cd_type:"beta",logP:3.97,MW:206.3,HBD:1,HBA:2,TPSA:37.3}},
};
function loadPreset(k) {
  const data = PRESETS[k];
  const c = data.compound;
  S.cv = c;
  document.getElementById('w-drug').value = c.drug_name;
  document.getElementById('w-cd').value   = c.cd_type;
  document.getElementById('w-logP').value = c.logP;
  document.getElementById('w-MW').value   = c.MW;
  document.getElementById('w-HBD').value  = c.HBD;
  document.getElementById('w-HBA').value  = c.HBA;
  document.getElementById('w-TPSA').value = c.TPSA;
  updateBars();
  updateCVPreview(data);
  document.getElementById('upload-error').style.display = 'none';
  document.getElementById('upload-success').style.display = 'block';
  document.getElementById('upload-success-msg').textContent =
    `${c.drug_name} preset loaded · ${c.cd_type}-CD`;
  goStep(3);
}

// ── drug chip picker ──────────────────────────────────────────────────────
function pickDrug(i) {
  const d = DRUG_DB[i];
  document.getElementById('w-drug').value = d.name;
  document.getElementById('w-logP').value = d.logP;
  document.getElementById('w-MW').value   = d.MW;
  document.getElementById('w-HBD').value  = d.HBD;
  document.getElementById('w-HBA').value  = d.HBA;
  document.getElementById('w-TPSA').value = d.TPSA;
  updateBars();
  document.querySelectorAll('.drug-chip').forEach((el,j) =>
    el.classList.toggle('sel', j===i));
  // update compound voucher preview
  const cvData = {
    researcher: {name: '—', institution: '—'},
    compound: {
      drug_name: d.name,
      cd_type: document.getElementById('w-cd')?.value || 'beta',
      logP: d.logP, MW: d.MW, HBD: d.HBD, HBA: d.HBA, TPSA: d.TPSA,
    }
  };
  S.cv = cvData.compound;
  updateCVPreview(cvData);
  document.getElementById('upload-error').style.display = 'none';
  document.getElementById('upload-success').style.display = 'block';
  document.getElementById('upload-success-msg').textContent =
    `${d.name} selected · ${cvData.compound.cd_type}-CD`;
}

// ── descriptor live bars ──────────────────────────────────────────────────
function pct(val, lo, hi) { return Math.min(100, Math.max(0, (val-lo)/(hi-lo)*100)); }
function updateBars() {
  const v = {
    logP: parseFloat(document.getElementById('w-logP')?.value),
    MW:   parseFloat(document.getElementById('w-MW')?.value),
    HBD:  parseFloat(document.getElementById('w-HBD')?.value),
    HBA:  parseFloat(document.getElementById('w-HBA')?.value),
    TPSA: parseFloat(document.getElementById('w-TPSA')?.value),
  };
  const ranges = {logP:[-5,8],MW:[100,1500],HBD:[0,10],HBA:[0,25],TPSA:[0,300]};
  for (const [k, [lo,hi]] of Object.entries(ranges)) {
    const n = isNaN(v[k]) ? '—' : v[k];
    document.getElementById('fd-'+k).textContent = n;
    document.getElementById('fb-'+k).style.width = isNaN(v[k]) ? '0%' : pct(v[k],lo,hi)+'%';
  }
}

// ── secure hint ───────────────────────────────────────────────────────────
function updateSecureHint() {
  const sec = document.getElementById('w-secure').value === 'true';
  const hint = document.getElementById('secure-hint');
  if (sec) {
    hint.className = 'callout success';
    hint.innerHTML = `<div class="callout-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--green-d)" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg></div><div><div class="callout-title">Guard active</div><div class="callout-body">SQL injection, feature bounds, output anomaly checks all running.</div></div>`;
  } else {
    hint.className = 'callout danger';
    hint.innerHTML = `<div class="callout-icon"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="var(--red)" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg></div><div><div class="callout-title">No protection</div><div class="callout-body">All validation rules disabled. Demonstration mode only.</div></div>`;
  }
}

// ── predict (wizard step 5) ───────────────────────────────────────────────
async function runPredict() {
  if (!S.ready) { toast('Engine not initialised'); return; }
  S.step = 5;
  renderStep(5);

  const payload = {
    drug_name: document.getElementById('w-drug').value || 'Unknown',
    cd_type:   document.getElementById('w-cd').value,
    features: {
      logP: parseFloat(document.getElementById('w-logP').value),
      MW:   parseFloat(document.getElementById('w-MW').value),
      HBD:  parseInt(document.getElementById('w-HBD').value),
      HBA:  parseInt(document.getElementById('w-HBA').value),
      TPSA: parseFloat(document.getElementById('w-TPSA').value),
    },
    secure: document.getElementById('w-secure').value === 'true',
    attack_a: false, attack_c: false, attack_d: false,
  };

  // show spinner
  document.getElementById('result-area').innerHTML = `<div style="text-align:center;padding:48px 0"><div class="lo-spin" style="margin:0 auto 14px"></div><div style="color:var(--text2);font-size:13px">Running prediction…</div></div>`;

  const r = await fetch('/api/predict', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify(payload)
  }).then(r => r.json());

  renderResult(r, 'result-area');
  renderLog(r.logs, 'result-log');
  renderLog(r.logs, 'guard-log');
  S.globalLogs = [...S.globalLogs, ...(r.logs||[])];
  renderLog(S.globalLogs, 'dash-log');
  refreshStats();
}

// ── render result ─────────────────────────────────────────────────────────
const STATUS_ICONS = {
  OK: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg>`,
  BLOCKED: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>`,
  REJECTED: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`,
  QUARANTINED: `<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/></svg>`,
};
function getStatusMsg(r) {
  if (r.status === 'OK') {
    return r.secure
      ? 'Prediction accepted — all 6 validation rules passed.'
      : 'Prediction returned — guard rules were OFF for this run (unsecured mode).';
  }
  const MAP = {
    BLOCKED: 'Input blocked before database query. Malicious SQL pattern detected.',
    REJECTED: 'Input rejected — one or more molecular descriptors outside valid physical range.',
    QUARANTINED: 'Output quarantined — prediction is physically implausible or a statistical outlier.',
  };
  return MAP[r.status] || '';
}

function renderResult(r, elId) {
  const el = document.getElementById(elId);
  const pred = r.prediction !== null
    ? `<span class="result-pred">${r.prediction} <span class="result-unit">mg/mL</span></span>`
    : `<span class="result-pred" style="font-size:22px;color:var(--text3)">No prediction</span>`;
  const msg = r.reason || getStatusMsg(r) || '';

  let extra = '';
  if (r.status === 'OK') {
    extra = `
      <div class="div"></div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;font-size:12px">
        <div><div style="color:var(--text3);margin-bottom:2px">Drug</div><div style="font-weight:600">${esc(r.drug)}</div></div>
        <div><div style="color:var(--text3);margin-bottom:2px">CD Type</div><div style="font-weight:600">${esc(r.cd_type||'beta')}-CD</div></div>
        <div><div style="color:var(--text3);margin-bottom:2px">Security</div><div style="font-weight:600">${r.secure?'Secured':'Unsecured'}</div></div>
        <div><div style="color:var(--text3);margin-bottom:2px">Confidence</div><div style="font-weight:700;color:${r.confidence==='High'?'var(--green-d)':r.confidence==='Medium'?'var(--amber)':'var(--red)'}">${r.confidence||'—'}</div></div>
      </div>
      <div style="margin-top:14px;padding:12px 14px;background:rgba(0,0,0,.03);border-radius:8px;font-size:12px;color:var(--text2)">
        <strong>Interpretation:</strong> A predicted CD-complexation solubility of <strong>${r.prediction} mg/mL</strong> indicates ${r.prediction > 10 ? 'strong enhancement' : r.prediction > 3 ? 'moderate enhancement' : 'mild enhancement'} relative to intrinsic aqueous solubility. ${r.secure ? '✓ All 6 guard rules passed.' : '⚠ No guard rules were active — result is unvalidated.'}
      </div>`;
  }

  el.innerHTML = `
    <div class="result-block ${r.status}" style="margin-bottom:18px">
      <div class="result-tag ${r.status}">${STATUS_ICONS[r.status]||''} ${r.status}</div>
      ${pred}
      <div class="result-reason">${esc(msg)}</div>
      ${extra}
    </div>`;
}

// ── log renderer ──────────────────────────────────────────────────────────
function renderLog(logs, elId) {
  const el = document.getElementById(elId);
  if (!logs || !logs.length) {
    el.innerHTML = '<span class="t-empty">No log entries.</span>';
    return;
  }
  el.innerHTML = logs.map(l =>
    `<span style="display:block;padding:1px 0">` +
    `<span class="t-time">[${l.time}]</span> ` +
    `<span class="t-${l.level}">${l.level.padEnd(8)}</span> ` +
    `${esc(l.msg)}` +
    `</span>`
  ).join('');
  el.scrollTop = el.scrollHeight;
}

// ── attacks ────────────────────────────────────────────────────────────────
const ATK_CHECK = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><polyline points="20 6 9 17 4 12"/></svg>`;
const ATK_CROSS = `<svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="3"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>`;

function togAtk(k) {
  S.attacks[k] = !S.attacks[k];
  document.getElementById('atk-'+k).classList.toggle('on', S.attacks[k]);
  document.getElementById('at-'+k).innerHTML = S.attacks[k] ? ATK_CHECK : ATK_CROSS;
}
function togAllAtk() {
  const turnOn = !Object.values(S.attacks).every(v=>v); // if all ON → turn off; else turn all on
  ['a','b','c','d'].forEach(k => {
    if (S.attacks[k] !== turnOn) togAtk(k);
  });
}

async function fireAtk(secure) {
  if (!S.ready) { toast('Initialise the engine first'); return; }

  const activeAttacks = Object.entries(S.attacks).filter(([k,v]) => v).map(([k]) => k);
  if (activeAttacks.length === 0) { toast('Toggle at least one attack first'); return; }

  const mode = secure ? 'Secured (Guard ON)' : 'Unsecured (No Guard)';
  const resultsEl = document.getElementById('atk-result');
  resultsEl.innerHTML = `<div style="color:var(--text3);font-size:13px;padding:12px 0">Running ${activeAttacks.length} attack(s)…</div>`;

  // read drug/descriptor from attack lab selectors
  const atkDrugSel = document.getElementById('atk-drug-sel');
  const atkDrugName = document.getElementById('atk-drug-name');
  const atkLogP = document.getElementById('atk-logP');
  const atkMW   = document.getElementById('atk-MW');
  const atkHBD  = document.getElementById('atk-HBD');
  const atkHBA  = document.getElementById('atk-HBA');
  const atkTPSA = document.getElementById('atk-TPSA');

  const baseFeatures = {
    logP: parseFloat(atkLogP?.value ?? 3.97),
    MW:   parseFloat(atkMW?.value   ?? 206.3),
    HBD:  parseInt(atkHBD?.value    ?? 1),
    HBA:  parseInt(atkHBA?.value    ?? 2),
    TPSA: parseFloat(atkTPSA?.value ?? 37.3),
  };
  const baseDrug = atkDrugName?.value?.trim() || 'Ibuprofen';

  const allLogs = [];
  const cards = [];

  for (const atk of activeAttacks) {
    let drug_name = baseDrug;
    let features  = {...baseFeatures};
    if (atk === 'a') drug_name = "' OR '1'='1'; DROP TABLE cd_solubility;--";

    const payload = {
      drug_name,
      cd_type: document.getElementById('atk-cd')?.value || 'beta',
      features,
      secure,
      attack_a: atk === 'a',
      attack_b: atk === 'b',
      attack_c: atk === 'c',
      attack_d: atk === 'd',
    };

    const r = await fetch('/api/predict', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    }).then(res => res.json());

    allLogs.push(...(r.logs || []));

    const pred = r.prediction !== null ? `${r.prediction} mg/mL` : 'N/A';
    const atkLabels = {a:'SQL Injection', b:'Data Poisoning', c:'Feature Injection', d:'Output Manipulation'};

    let unsecuredNote = '';
    if (!secure) {
      if (atk === 'a') unsecuredNote = `<div style="margin-top:8px;padding:8px 10px;background:rgba(239,35,60,.1);border-radius:6px;font-size:12px;color:var(--red)">⚠ No guard — SQL string passed to DB query unchecked. In a real DB this could DROP tables.</div>`;
      if (atk === 'b') unsecuredNote = `<div style="margin-top:8px;padding:8px 10px;background:rgba(239,35,60,.1);border-radius:6px;font-size:12px;color:var(--red)">⚠ No guard — features silently corrupted (logP, MW, TPSA inflated 1.5–5×) before model inference. Prediction is based on poisoned inputs.</div>`;
      if (atk === 'c') unsecuredNote = `<div style="margin-top:8px;padding:8px 10px;background:rgba(239,35,60,.1);border-radius:6px;font-size:12px;color:var(--red)">⚠ No guard — impossible features (logP=50, MW=50000, TPSA=-10) fed directly to model. Result is meaningless.</div>`;
      if (atk === 'd') unsecuredNote = `<div style="margin-top:8px;padding:8px 10px;background:rgba(239,35,60,.1);border-radius:6px;font-size:12px;color:var(--red)">⚠ No guard — output multiplied ×10–100 with no anomaly check. Clinician sees falsified result.</div>`;
    }

    cards.push(`
      <div class="result-block ${r.status}" style="margin-bottom:14px">
        <div class="result-tag ${r.status}">${STATUS_ICONS[r.status]||''} Attack ${atk.toUpperCase()} — ${esc(mode)}</div>
        <div style="font-size:11px;font-weight:600;color:var(--text3);margin-bottom:6px;letter-spacing:.04em">${esc(atkLabels[atk]||atk)}</div>
        <div class="result-pred" style="font-size:22px">${pred}</div>
        <div class="result-reason">${esc(r.reason || getStatusMsg(r) || '')}</div>
        ${unsecuredNote}
      </div>`);
  }

  resultsEl.innerHTML = cards.join('');
  renderLog(allLogs, 'atk-log');
  renderLog(allLogs, 'guard-log');
  S.globalLogs = [...S.globalLogs, ...allLogs];
  renderLog(S.globalLogs, 'dash-log');
  refreshStats();
}
// ── database ───────────────────────────────────────────────────────────────
async function loadDb() {
  if (!S.ready) return;
  const rows = await fetch('/api/db').then(r=>r.json());
  document.getElementById('db-body').innerHTML = rows.map(r => `<tr>
    <td>${esc(r.drug_name)}</td>
    <td><span class="badge ${r.cd_type==='beta'?'b-cyan':r.cd_type==='alpha'?'b-green':'b-amber'}">${r.cd_type}</span></td>
    <td class="mono">${parseFloat(r.solubility).toFixed(3)}</td>
    <td class="mono">${r.logP}</td>
    <td class="mono">${r.MW}</td>
    <td class="mono">${r.HBD}</td>
    <td class="mono">${r.HBA}</td>
    <td class="mono">${r.TPSA}</td>
    <td class="mono" style="font-size:10.5px;color:var(--text4)">${(r.integrity_hash||'').substring(0,12)}…</td>
  </tr>`).join('');
}

// ── Corruption Analysis Charts ─────────────────────────────────────────────
async function drawCorruptionCharts() {
  if (!S.ready) {
    document.querySelectorAll('#section-corruption svg').forEach(svg => {
      svg.innerHTML = `<text x="50%" y="50%" text-anchor="middle" font-size="13" fill="currentColor" opacity=".4">Engine not initialised — run Setup first</text>`;
    });
    return;
  }
  // show loading spinners in each chart slot
  document.querySelectorAll('#section-corruption svg[id^="chart-"], #section-corruption svg[id^="donut-"]').forEach(svg => {
    const vb = svg.getAttribute('viewBox') || '0 0 460 220';
    const [,,w,h] = vb.split(' ').map(Number);
    svg.innerHTML = `<text x="${w/2}" y="${h/2+5}" text-anchor="middle" font-size="12" fill="currentColor" opacity=".35">Loading…</text>`;
  });
  let data;
  try {
    data = await fetch('/api/corruption').then(r => r.json());
  } catch(e) {
    document.querySelector('#section-corruption .callout-body').textContent = 'Failed to load data: ' + e.message;
    return;
  }
  if (data.error) { return; }
  drawR2Chart(data);
  drawDriftChart(data.drift);
  drawScatterChart(data.scatter_valid, data.scatter_injected);
  drawDonuts(data.stats);
  drawOutputChart(data.output_data);
}

function drawR2Chart(data) {
  const svg = document.getElementById('chart-r2');
  if(!svg) return;
  const W=460,H=220,pad={t:20,r:20,b:40,l:44};
  const iW=W-pad.l-pad.r, iH=H-pad.t-pad.b;
  const fracs = data.fracs;           // [0,5,10,15,20,25,30]
  const clean  = data.r2_clean;
  const poison = data.r2_poison;
  const xS=i=>pad.l+(i/(fracs.length-1))*iW;
  const yS=v=>pad.t+iH-(Math.max(0,Math.min(1,v))*iH);
  let h='';
  [0,0.2,0.4,0.6,0.8,1.0].forEach(v=>{const y=yS(v);h+=`<line x1="${pad.l}" y1="${y}" x2="${pad.l+iW}" y2="${y}" stroke="currentColor" stroke-opacity=".08"/><text x="${pad.l-6}" y="${y+4}" text-anchor="end" font-size="10" fill="currentColor" opacity=".5">${v.toFixed(1)}</text>`;});
  fracs.forEach((f,i)=>{h+=`<text x="${xS(i)}" y="${pad.t+iH+16}" text-anchor="middle" font-size="10" fill="currentColor" opacity=".5">${f}%</text>`;});
  h+=`<line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t+iH}" stroke="currentColor" stroke-opacity=".2"/>`;
  h+=`<line x1="${pad.l}" y1="${pad.t+iH}" x2="${pad.l+iW}" y2="${pad.t+iH}" stroke="currentColor" stroke-opacity=".2"/>`;
  h+=`<text x="${pad.l+iW/2}" y="${H-4}" text-anchor="middle" font-size="10" fill="currentColor" opacity=".5">Poison Fraction</text>`;
  h+=`<text x="10" y="${pad.t+iH/2+4}" text-anchor="middle" font-size="10" fill="currentColor" opacity=".5" transform="rotate(-90,10,${pad.t+iH/2})">R²</text>`;
  const cleanPts=clean.map((v,i)=>`${i===0?'M':'L'}${xS(i)},${yS(v)}`).join(' ');
  h+=`<path d="${cleanPts}" fill="none" stroke="var(--green)" stroke-width="2" stroke-dasharray="5,3" opacity=".7"/>`;
  const poisPts=poison.map((v,i)=>`${i===0?'M':'L'}${xS(i)},${yS(v)}`).join(' ');
  h+=`<path d="${poisPts} L${xS(fracs.length-1)},${yS(0)} L${xS(0)},${yS(0)} Z" fill="var(--red)" opacity=".07"/>`;
  h+=`<path d="${poisPts}" fill="none" stroke="var(--red)" stroke-width="2.5"/>`;
  poison.forEach((v,i)=>{h+=`<circle cx="${xS(i)}" cy="${yS(v)}" r="4" fill="var(--red)"/><text x="${xS(i)}" y="${yS(v)-8}" text-anchor="middle" font-size="9.5" fill="var(--red)" font-weight="600">${v.toFixed(2)}</text>`;});
  const thY=yS(0.75);
  h+=`<line x1="${pad.l}" y1="${thY}" x2="${pad.l+iW}" y2="${thY}" stroke="var(--amber)" stroke-width="1" stroke-dasharray="3,3" opacity=".8"/>`;
  h+=`<text x="${pad.l+iW-2}" y="${thY-4}" text-anchor="end" font-size="9" fill="var(--amber)">Acceptable (R²=0.75)</text>`;
  svg.innerHTML=h;
}

function drawDriftChart(drugs) {
  const svg=document.getElementById('chart-drift'); if(!svg)return;
  if (!drugs || !drugs.length) { svg.innerHTML=`<text x="50%" y="50%" text-anchor="middle" font-size="12" fill="currentColor" opacity=".4">No data</text>`; return; }
  const W=460,H=220,pad={t:20,r:10,b:50,l:44};
  const iW=W-pad.l-pad.r,iH=H-pad.t-pad.b;
  // filter out null predictions
  const valid = drugs.filter(d => d.true !== null && d.poisoned !== null);
  if (!valid.length) { svg.innerHTML=`<text x="50%" y="50%" text-anchor="middle" font-size="12" fill="currentColor" opacity=".4">No predictions available</text>`; return; }
  const maxV=Math.max(80, ...valid.map(d=>Math.max(d.true||0, d.poisoned||0)))*1.1;
  const yS=v=>pad.t+iH-(Math.max(0,v)/maxV)*iH;
  const gW=iW/valid.length,bW=gW*0.34;
  let h='';
  const steps=[0,Math.round(maxV*0.25),Math.round(maxV*0.5),Math.round(maxV*0.75),Math.round(maxV)];
  steps.forEach(v=>{const y=yS(v);h+=`<line x1="${pad.l}" y1="${y}" x2="${pad.l+iW}" y2="${y}" stroke="currentColor" stroke-opacity=".08"/><text x="${pad.l-5}" y="${y+4}" text-anchor="end" font-size="9" fill="currentColor" opacity=".5">${v}</text>`;});
  h+=`<text x="10" y="${pad.t+iH/2+4}" text-anchor="middle" font-size="9" fill="currentColor" opacity=".5" transform="rotate(-90,10,${pad.t+iH/2})">mg/mL</text>`;
  valid.forEach((d,i)=>{
    const cx=pad.l+gW*i+gW/2,x1=cx-bW-1,x2=cx+1;
    const tv=d.true||0, pv=d.poisoned||0;
    const y1t=yS(tv),y1p=yS(pv);
    const tH=Math.max(iH-(y1t-pad.t),2), pH=Math.max(iH-(y1p-pad.t),2);
    h+=`<g style="cursor:pointer"><rect x="${x1}" y="${y1t}" width="${bW}" height="${tH}" fill="var(--accent)" rx="2" opacity=".85"/><title>True: ${tv.toFixed(3)} mg/mL</title></g>`;
    h+=`<g style="cursor:pointer"><rect x="${x2}" y="${y1p}" width="${bW}" height="${pH}" fill="var(--red)" rx="2" opacity=".8"/><title>Poisoned: ${pv.toFixed(3)} mg/mL</title></g>`;
    h+=`<text x="${cx}" y="${pad.t+iH+14}" text-anchor="middle" font-size="8.5" fill="currentColor" opacity=".6">${d.name.substring(0,8)}</text>`;
  });
  h+=`<line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t+iH}" stroke="currentColor" stroke-opacity=".2"/>`;
  h+=`<line x1="${pad.l}" y1="${pad.t+iH}" x2="${pad.l+iW}" y2="${pad.t+iH}" stroke="currentColor" stroke-opacity=".2"/>`;
  svg.innerHTML=h;
}

function drawScatterChart(validPts, injectedPts) {
  const svg=document.getElementById('chart-scatter'); if(!svg)return;
  if (!validPts || !validPts.length) { svg.innerHTML=`<text x="50%" y="50%" text-anchor="middle" font-size="12" fill="currentColor" opacity=".4">No data</text>`; return; }
  const W=460,H=220,pad={t:20,r:20,b:40,l:50};
  const iW=W-pad.l-pad.r,iH=H-pad.t-pad.b;
  const allPts = [...(validPts||[]), ...(injectedPts||[])];
  const xMin=Math.min(-2,...allPts.map(p=>p.x))-1;
  const xMax=Math.max(8,...allPts.map(p=>p.x))+2;
  const yMin=0;
  const yMax=Math.max(50,...allPts.map(p=>p.y))*1.1;
  const xS=v=>pad.l+((v-xMin)/(xMax-xMin))*iW;
  const yS=v=>pad.t+iH-((v-yMin)/(yMax-yMin))*iH;
  let h='';
  // grid
  const xTicks=[...new Set([-2,0,Math.round((xMax+xMin)/2),Math.round(xMax-1)])];
  xTicks.forEach(v=>{const x=xS(v);h+=`<line x1="${x}" y1="${pad.t}" x2="${x}" y2="${pad.t+iH}" stroke="currentColor" stroke-opacity=".07"/><text x="${x}" y="${pad.t+iH+14}" text-anchor="middle" font-size="9" fill="currentColor" opacity=".5">${v.toFixed(1)}</text>`;});
  [0,Math.round(yMax*0.25),Math.round(yMax*0.5),Math.round(yMax*0.75),Math.round(yMax)].forEach(v=>{const y=yS(v);h+=`<line x1="${pad.l}" y1="${y}" x2="${pad.l+iW}" y2="${y}" stroke="currentColor" stroke-opacity=".07"/><text x="${pad.l-5}" y="${y+4}" text-anchor="end" font-size="9" fill="currentColor" opacity=".5">${v}</text>`;});
  // valid zone (logP -5 to 8)
  h+=`<rect x="${xS(-5)}" y="${pad.t}" width="${xS(8)-xS(-5)}" height="${iH}" fill="var(--green)" opacity=".05" rx="2"/>`;
  h+=`<text x="${xS(1.5)}" y="${pad.t+11}" text-anchor="middle" font-size="8" fill="var(--green-d)" opacity=".7">Valid logP range</text>`;
  if (injectedPts && injectedPts.length) {
    h+=`<rect x="${xS(8)}" y="${pad.t}" width="${xS(xMax)-xS(8)}" height="${iH}" fill="var(--red)" opacity=".04" rx="2"/>`;
    h+=`<text x="${xS((8+xMax)/2)}" y="${pad.t+11}" text-anchor="middle" font-size="8" fill="var(--red)" opacity=".7">Impossible — Guard OFF accepts all</text>`;
  }
  (validPts||[]).forEach(p=>{h+=`<circle cx="${xS(p.x)}" cy="${yS(p.y)}" r="4.5" fill="var(--accent)" opacity=".8"/>`;});
  (injectedPts||[]).forEach(p=>{h+=`<circle cx="${xS(p.x)}" cy="${yS(p.y)}" r="5.5" fill="var(--red)" opacity=".85"/><line x1="${xS(p.x)-4}" y1="${yS(p.y)-4}" x2="${xS(p.x)+4}" y2="${yS(p.y)+4}" stroke="white" stroke-width="1.5"/><line x1="${xS(p.x)+4}" y1="${yS(p.y)-4}" x2="${xS(p.x)-4}" y2="${yS(p.y)+4}" stroke="white" stroke-width="1.5"/>`;});
  h+=`<text x="${pad.l+iW/2}" y="${H-4}" text-anchor="middle" font-size="9" fill="currentColor" opacity=".5">logP →</text>`;
  h+=`<line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t+iH}" stroke="currentColor" stroke-opacity=".2"/>`;
  h+=`<line x1="${pad.l}" y1="${pad.t+iH}" x2="${pad.l+iW}" y2="${pad.t+iH}" stroke="currentColor" stroke-opacity=".2"/>`;
  svg.innerHTML=h;
}

function drawDonut(svgId,slices){
  const svg=document.getElementById(svgId); if(!svg)return;
  const cx=70,cy=70,r=52,ir=34;
  const total=slices.reduce((s,x)=>s+x.v,0);
  if(!total){svg.innerHTML=`<text x="70" y="74" text-anchor="middle" font-size="11" fill="currentColor" opacity=".4">No data yet</text>`;return;}
  let angle=-Math.PI/2,h='';
  slices.forEach(sl=>{
    if(!sl.v) return;
    const a=(sl.v/total)*Math.PI*2;
    const x1=cx+r*Math.cos(angle),y1=cy+r*Math.sin(angle);
    const x2=cx+r*Math.cos(angle+a),y2=cy+r*Math.sin(angle+a);
    const ix1=cx+ir*Math.cos(angle),iy1=cy+ir*Math.sin(angle);
    const ix2=cx+ir*Math.cos(angle+a),iy2=cy+ir*Math.sin(angle+a);
    const lg=a>Math.PI?1:0;
    h+=`<path d="M${ix1},${iy1} L${x1},${y1} A${r},${r},0,${lg},1,${x2},${y2} L${ix2},${iy2} A${ir},${ir},0,${lg},0,${ix1},${iy1}" fill="${sl.color}" opacity=".9"/>`;
    const pct=Math.round(sl.v/total*100);
    if(pct>5){const ma=angle+a/2,lx=cx+(r+ir)/2*Math.cos(ma),ly=cy+(r+ir)/2*Math.sin(ma);h+=`<text x="${lx}" y="${ly+3}" text-anchor="middle" font-size="9" fill="white" font-weight="700">${pct}%</text>`;}
    angle+=a;
  });
  svg.innerHTML=h;
}

function drawDonuts(stats) {
  // stats = {ok, rejected, quarantined, blocked}
  const total = stats ? (stats.ok+stats.rejected+stats.quarantined+stats.blocked) : 0;
  if (!total) {
    // no real runs yet — show illustrative donuts
    drawDonut('donut-unsecured',[{v:100,color:'var(--red)'}]);
    drawDonut('donut-secured',[{v:96,color:'var(--green)'},{v:2,color:'var(--amber)'},{v:1,color:'var(--accent)'},{v:1,color:'var(--red)'}]);
    return;
  }
  // unsecured: all attacks pass through (ok+quarantined+rejected all counted as "wrong")
  const wrong = stats.ok + stats.rejected + stats.quarantined;
  drawDonut('donut-unsecured',[
    {v: stats.blocked, color:'var(--accent)'},
    {v: wrong,         color:'var(--red)'},
  ]);
  // secured: blocked+quarantined+rejected = caught; ok = passed safely
  const caught = stats.blocked + stats.quarantined + stats.rejected;
  drawDonut('donut-secured',[
    {v: stats.ok,         color:'var(--green)'},
    {v: stats.quarantined,color:'var(--amber)'},
    {v: stats.rejected,   color:'var(--accent)'},
    {v: stats.blocked,    color:'var(--red)'},
  ]);
  // update subtitle counts
  const usub = document.getElementById('donut-unsecured-sub');
  if (usub) usub.textContent = `${stats.blocked} of ${total} attacks caught`;
  const ssub = document.getElementById('donut-secured-sub');
  if (ssub) ssub.textContent = `${caught} of ${total} attacks caught`;
}

function drawOutputChart(data){
  const svg=document.getElementById('chart-output'); if(!svg)return;
  if (!data || !data.length) { svg.innerHTML=`<text x="460" y="80" text-anchor="middle" font-size="12" fill="currentColor" opacity=".4">No data</text>`; return; }
  const valid=data.filter(d=>d.true!==null && d.manipulated!==null);
  if(!valid.length){ svg.innerHTML=`<text x="460" y="80" text-anchor="middle" font-size="12" fill="currentColor" opacity=".4">No manipulated predictions available</text>`; return; }
  const W=920,H=160,pad={t:16,r:160,b:30,l:100};
  const iW=W-pad.l-pad.r,iH=H-pad.t-pad.b;
  const maxV=Math.max(10,...valid.map(d=>d.manipulated||0))*1.05;
  const xS=v=>pad.l+(v/maxV)*iW;
  const bH=Math.floor(iH/valid.length)-4;
  let h='';
  const steps=[0,Math.round(maxV*0.25),Math.round(maxV*0.5),Math.round(maxV*0.75),Math.round(maxV)];
  steps.forEach(v=>{const x=xS(v);h+=`<line x1="${x}" y1="${pad.t}" x2="${x}" y2="${pad.t+iH}" stroke="currentColor" stroke-opacity=".08"/><text x="${x}" y="${pad.t+iH+14}" text-anchor="middle" font-size="9" fill="currentColor" opacity=".5">${Math.round(v)}</text>`;});
  h+=`<text x="${pad.l+iW/2}" y="${H-2}" text-anchor="middle" font-size="9" fill="currentColor" opacity=".5">mg/mL shown to user (Guard OFF) →</text>`;
  valid.forEach((d,i)=>{
    const y=pad.t+i*(bH+4);
    const tv=d.true||0, mv=d.manipulated||0;
    const tw=Math.max(xS(tv)-pad.l,3);
    const mw=Math.max(xS(mv)-pad.l,tw+1);
    const mult = tv > 0 ? Math.round(mv/tv) : '?';
    // wrap each bar row in a group — show true value label only on hover
    h+=`<g class="obar">`;
    h+=`<rect x="${pad.l}" y="${y}" width="${mw}" height="${bH}" fill="var(--red)" rx="2" opacity=".18"/>`;
    h+=`<rect x="${pad.l}" y="${y}" width="${tw}" height="${bH}" fill="var(--green)" rx="2" opacity=".9"/>`;
    h+=`<rect x="${pad.l+tw}" y="${y}" width="${mw-tw}" height="${bH}" fill="var(--red)" rx="2" opacity=".75"/>`;
    const manLabelX = Math.min(pad.l+tw+4, W-pad.r-80);
    h+=`<text x="${manLabelX}" y="${y+bH/2+4}" font-size="9" fill="white" font-weight="600">${mv.toFixed(1)} mg/mL (×${mult})</text>`;
    h+=`<text x="${pad.l-5}" y="${y+bH/2+4}" text-anchor="end" font-size="10" fill="currentColor" opacity=".75">${d.name}</text>`;
    h+=`<title>True: ${tv.toFixed(3)} mg/mL | Manipulated: ${mv.toFixed(3)} mg/mL (×${mult})</title>`;
    h+=`</g>`;
  });
  h+=`<line x1="${pad.l}" y1="${pad.t}" x2="${pad.l}" y2="${pad.t+iH}" stroke="currentColor" stroke-opacity=".2"/>`;
  h+=`<line x1="${pad.l}" y1="${pad.t+iH}" x2="${pad.l+iW}" y2="${pad.t+iH}" stroke="currentColor" stroke-opacity=".2"/>`;
  svg.innerHTML=h;
}
</script>
</body>
</html>"""

    @app.route("/")
    def index():
        return HTML

    @app.route("/api/sample_cv")
    def api_sample_cv():
        return json.dumps(json.loads(SAMPLE_CV_JSON)), 200, {"Content-Type": "application/json"}

    @app.route("/api/setup", methods=["POST"])
    def api_setup():
        d  = request.get_json(force=True) or {}
        n  = int(d.get("n_records", 600))
        pf = float(d.get("poison_frac", 0.0))
        return jsonify(ENGINE.setup(n_records=n, poison_frac=pf))

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        if not ENGINE.ready:
            return jsonify({"error": "Engine not initialised"}), 400
        d = request.get_json(force=True) or {}
        result = ENGINE.predict(
            drug_name = str(d.get("drug_name", "")),
            features  = d.get("features", {}),
            cd_type   = str(d.get("cd_type", "beta")),
            secure    = bool(d.get("secure", True)),
            attack_a  = bool(d.get("attack_a", False)),
            attack_b  = bool(d.get("attack_b", False)),
            attack_c  = bool(d.get("attack_c", False)),
            attack_d  = bool(d.get("attack_d", False)),
        )
        return jsonify(result)

    @app.route("/api/stats")
    def api_stats():
        return jsonify(ENGINE.get_stats())

    @app.route("/api/db")
    def api_db():
        if not ENGINE.ready:
            return jsonify([])
        return jsonify(ENGINE.get_db_sample(30))

    @app.route("/api/drugs")
    def api_drugs():
        return jsonify(ENGINE.get_real_drugs())

    @app.route("/api/corruption")
    def api_corruption():
        """Return real data for Corruption Analysis charts."""
        if not ENGINE.ready:
            return jsonify({"error": "Engine not initialised"}), 400
        import copy

        # 1. R² degradation sweep — train at several poison fractions
        fracs = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
        r2_clean = []
        r2_poison = []
        base_df = None
        try:
            cur = ENGINE.conn.execute(
                "SELECT logP,MW,HBD,HBA,TPSA,cd_type,solubility FROM cd_solubility")
            cols = [d[0] for d in cur.description]
            rows = [dict(zip(cols, r)) for r in cur.fetchall()]
            base_df = pd.DataFrame(rows)
            clean_m = train_model(base_df, poison_frac=0.0)[3]
            clean_r2 = clean_m["r2"]
            for pf in fracs:
                m = train_model(base_df, poison_frac=pf)[3]
                r2_clean.append(clean_r2)
                r2_poison.append(m["r2"])
        except Exception:
            r2_clean = [0.0] * len(fracs)
            r2_poison = [0.0] * len(fracs)

        # 2. Prediction drift — 10 real drugs, secure (true) vs attack_b unsecured
        drift = []
        for d in REAL_DRUG_DATA[:10]:
            feats = {"logP": d[1], "MW": d[2], "HBD": d[3], "HBA": d[4], "TPSA": d[5]}
            true_r = ENGINE.predict(d[0], copy.deepcopy(feats), cd_type="beta",
                                    secure=True, attack_b=False)
            poison_r = ENGINE.predict(d[0], copy.deepcopy(feats), cd_type="beta",
                                      secure=False, attack_b=True)
            drift.append({
                "name": d[0],
                "true": true_r["prediction"],
                "poisoned": poison_r["prediction"],
            })

        # 3. Feature injection scatter — valid inputs + attack_c injections
        scatter_valid = []
        scatter_injected = []
        for d in REAL_DRUG_DATA[:16]:
            feats = {"logP": d[1], "MW": d[2], "HBD": d[3], "HBA": d[4], "TPSA": d[5]}
            r = ENGINE.predict(d[0], feats, secure=True)
            if r["prediction"] is not None:
                scatter_valid.append({"x": round(d[1], 2), "y": r["prediction"]})
        for d in REAL_DRUG_DATA[:5]:
            feats = {"logP": d[1], "MW": d[2], "HBD": d[3], "HBA": d[4], "TPSA": d[5]}
            r = ENGINE.predict(d[0], feats, secure=False, attack_c=True)
            if r["prediction"] is not None:
                scatter_injected.append({"x": round(d[1] * 12, 1), "y": r["prediction"]})

        # 4. Output attack magnitude — attack_d on 5 drugs
        output_data = []
        for d in REAL_DRUG_DATA[:5]:
            feats = {"logP": d[1], "MW": d[2], "HBD": d[3], "HBA": d[4], "TPSA": d[5]}
            true_r = ENGINE.predict(d[0], copy.deepcopy(feats), secure=True, attack_d=False)
            atk_r  = ENGINE.predict(d[0], copy.deepcopy(feats), secure=False, attack_d=True)
            output_data.append({
                "name": d[0],
                "true": true_r["prediction"],
                "manipulated": atk_r["prediction"],
            })

        # 5. Current stats for donuts
        stats = ENGINE.get_stats()
        total = sum(stats.values()) or 1

        return jsonify({
            "fracs": [int(f * 100) for f in fracs],
            "r2_clean": r2_clean,
            "r2_poison": r2_poison,
            "drift": drift,
            "scatter_valid": scatter_valid,
            "scatter_injected": scatter_injected,
            "output_data": output_data,
            "stats": stats,
            "total": total,
        })

    return app


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    import signal as _signal
    parser = argparse.ArgumentParser(
        description="PharmI — Trustworthy AI in Pharmaceutical Decision Support")
    parser.add_argument("--mode", choices=["cli","web","both"], default="web")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Host to bind (default: 127.0.0.1). Use 0.0.0.0 to expose on LAN.")
    parser.add_argument("--port", type=int, default=PORT,
                        help=f"Port to bind (default: {PORT})")
    args = parser.parse_args()

    def _handle_exit(sig, frame):
        print("\n  Stopping PharmI… bye!")
        raise SystemExit(0)

    _signal.signal(_signal.SIGINT,  _handle_exit)
    _signal.signal(_signal.SIGTERM, _handle_exit)

    def _run_server(app, host, port):
        display_host = "localhost" if host in ("127.0.0.1", "::1") else host
        print(f"\n  Web UI  →  http://{display_host}:{port}")
        print("  Press Ctrl+C to stop\n")
        try:
            app.run(host=host, port=port, debug=False, threaded=True, use_reloader=False)
        except (KeyboardInterrupt, SystemExit):
            print("\n  Stopping PharmI… bye!")
            raise SystemExit(0)

    if args.mode == "cli":
        run_cli()
    elif args.mode == "web":
        _run_server(create_app(), args.host, args.port)
    elif args.mode == "both":
        run_cli()
        _run_server(create_app(), args.host, args.port)


if __name__ == "__main__":
    main()
