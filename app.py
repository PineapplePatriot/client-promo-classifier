"""
OKKO Classifier — Flask API server
===================================
Wraps okko_intervention_classifier_v2.py and exposes two endpoints:

  GET  /sample?n=20        → list of n random PER_IDs from the fuel files
  POST /classify           → { "per_id": "..." } → full profile + verdict JSON

Run:
    pip install flask flask-cors
    python app.py

Then open okko_visualizer.html in your browser.
Place this file in the same folder as okko_intervention_classifier_v2.py
and your test_data/ / data/ directories.
"""

import random
import importlib.util
import sys
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Load classifier module from same directory ──────────────
spec = importlib.util.spec_from_file_location(
    "classifier",
    os.path.join(os.path.dirname(__file__), "okko_intervention_classifier_v2.py"),
)
clf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(clf)

app = Flask(__name__)
CORS(app)

# Pre-load data once on startup
print("Loading data files...")
try:
    DATA = clf.load_all_data(verbose=True)
    ALL_IDS = sorted(
        set(DATA["fuel_main"]["PER_ID"]) | set(DATA["fuel_march"]["PER_ID"])
    )
    print(f"Ready. {len(ALL_IDS):,} unique customer IDs loaded.")
except Exception as e:
    print(f"ERROR loading data: {e}")
    DATA = None
    ALL_IDS = []


def _safe(val):
    """Convert numpy/pandas scalars to JSON-serialisable Python types."""
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return float(val)
    if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
        return None
    if val is pd.NaT:
        return None
    if isinstance(val, pd.Timestamp):
        return val.strftime("%d.%m.%Y") if pd.notna(val) else None
    if isinstance(val, dict):
        return {k: _safe(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [_safe(v) for v in val]
    return val


def profile_to_dict(p):
    """Flatten the profile dict into JSON-safe structure."""
    if not p.get("found"):
        return {"found": False}

    pre  = p.get("pre",  {})
    post = p.get("post", {})
    nf_pre  = p.get("nf_pre",  {})
    nf_post = p.get("nf_post", {})

    return {
        "found":            True,
        "inactive":         bool(p.get("inactive", False)),
        "last_date":        _safe(p.get("last_date")),
        # Fuel
        "p100_status":      p.get("p100_status", "never"),
        "p100_share_tier":  p.get("p100_share_tier", "none"),
        "p100_share_alltime": _safe(p.get("p100_share_alltime", 0.0)),
        "last_p100_date":   _safe(p.get("last_p100_date")),
        "avg_L_alltime":    _safe(p.get("avg_L_alltime")),
        "pre_fills":        _safe(pre.get("n", 0)),
        "pre_avg_L":        _safe(pre.get("avg_L")),
        "pre_median_L":     _safe(pre.get("median_L")),
        "pre_p100_n":       _safe(pre.get("p100_n", 0)),
        "pre_p100_share":   _safe(pre.get("p100_share", 0.0)),
        "march_fills":      _safe(post.get("n", 0)),
        "march_avg_L":      _safe(post.get("avg_L")),
        "march_p100":       bool(post.get("ever_p100", False)),
        # Non-fuel
        "buys_nonfuel":     bool(p.get("buys_nonfuel", False)),
        "buys_coffee":      bool(p.get("buys_coffee", False)),
        "nf_monthly_pre":   _safe(nf_pre.get("monthly_spend", 0)),
        "nf_monthly_march": _safe(nf_post.get("monthly_spend", 0)),
        "nf_change_pct":    _safe(p.get("nf_change_pct", 0.0)),
        "nf_dropped":       bool(p.get("nf_dropped", False)),
        # Partners
        "pharmacy_only":    bool(p.get("pharmacy_only", False)),
        "partner_names":    p.get("partner_names", []),
        # PP
        "pp_score":         _safe(p.get("pp_score", 0.0)),
        "pp_tercile":       p.get("pp_tercile", "low"),
    }


@app.route("/sample")
def sample():
    n = min(int(request.args.get("n", 20)), 100)
    if not ALL_IDS:
        return jsonify({"error": "Data not loaded"}), 500
    ids = random.sample(ALL_IDS, min(n, len(ALL_IDS)))
    return jsonify({"ids": ids})


@app.route("/classify", methods=["POST"])
def classify_endpoint():
    if DATA is None:
        return jsonify({"error": "Data not loaded"}), 500

    body = request.get_json(force=True)
    per_id = str(body.get("per_id", "")).strip()
    if not per_id:
        return jsonify({"error": "per_id is required"}), 400

    profile = clf.compute_profile(per_id, DATA)
    code, reason, flags = clf.classify(profile)

    return jsonify({
        "per_id":  per_id,
        "code":    code,
        "reason":  reason,
        "flags":   flags,
        "profile": profile_to_dict(profile),
    })


if __name__ == "__main__":
    app.run(debug=False, port=5050)
