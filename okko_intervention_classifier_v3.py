"""
OKKO Customer Intervention Classifier  v3
==========================================
Reads all transaction and enrichment files, computes a full behavioral
profile for a given customer ID, and recommends one of four interventions:

  A1  — Bundling offer (free coffee promo)
  A2  — Discount coupon (-3 UAH/L on P100)
  B1  — Educational push (P100 awareness)
  --  — Do not recommend

Decision tree (v2 — multi-factor, three-block):
  UNIVERSAL EXCLUSIONS first:
    • Inactive >2 months                       → --
    • Pharmacy-only partner spend              → --
  BLOCK 1 — ACTIVE P100 buyers:
    • Has non-fuel + nf dropped >20% in March  → A1  (Shapiro spike)
    • Share <20%  + has non-fuel               → A1  (deepen pairing)
    • Share 20-50% + has non-fuel              → A1  (reinforce share)
    • avg ≥30L (any share)                     → A2  (loyalty/volume)
    • Share 50-70% or 70%+                     → A2  (loyalty reward)
    • Else                                     → --
  BLOCK 2 — LAPSED P100 buyers:
    • avg ≥30L + has non-fuel                  → A1  (soft re-entry)
    • nf dropped >20% + has non-fuel           → A1  (spike-driven lapse)
    • high PP + has non-fuel                   → A1  (means + station habit)
    • avg ≥30L + no non-fuel                   → A2  (price re-entry)
    • high PP + no non-fuel                    → B1  (educate, no bundle hook)
    • Else                                     → --
  BLOCK 3 — NEVER P100:
    • avg ≥30L + has non-fuel                  → A1  (bundle entry)
    • high PP (any volume)                     → A2  (adoption barrier)
    • avg ≥30L + no non-fuel + mid PP          → A2  (volume signal)
    • avg ≥30L + no non-fuel + low PP          → B1  (educate first)
    • Else                                     → --

  Coffee-buyer flag enriches A1 reason text wherever assigned.
  buys_nonfuel requires ≥2 pre-period transactions (noise threshold).

Run:
    python okko_intervention_classifier_v3.py
    python okko_intervention_classifier_v3.py --id 12345678
    python okko_intervention_classifier_v3.py --id 12345678 --verbose

FOLDER STRUCTURE (same as Parts 1-3):
  test_data/   transaction CSVs
  data/        enrichment files
"""

import argparse
import sys
import os
import warnings

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# 0. CONFIG
# ─────────────────────────────────────────────────────────────
data_folder            = "test_data/"
enrichment_data_folder = "data/"

FUEL_FILE_MAIN  = data_folder + "trn_table_fuel_01_01_25_28_02_26.csv"
FUEL_FILE_MARCH = data_folder + "trn_table_fuel_01_03_26_22_03_26.csv"
NONFUEL_FILE    = data_folder + "trn_table_not_fuel_01_10_25_30_03_26.csv"
PARTNER_FILE    = data_folder + "trn_table_partners_01_10_25_31_03_26.csv"
PRODUCT_FUEL    = enrichment_data_folder + "product_table_fuel.xlsx"
PERSONAL_FILE   = enrichment_data_folder + "personal_info.csv"

PARTNER_ENCODING = "cp1251"
SEP              = ";"
PULLS_100_CODE   = 119620

PHARMACY_PARTNERS     = {"Аптека \"Подорожник\"", "Аптека Доброго Дня",
                         "Аптека \"БАМ\"", "Здорова родина", "Ощад аптека"}
RAIFFEISEN_PARTNERS   = {"Райффайзен Банк"}
RETAIL_DISC_PARTNERS  = {"АЛЛО", "Книголенд", "BlaBlaCar", "TOKA", "Веселка"}
INSURANCE_MED_PARTNERS = {"POLIS", "SmartLab", "МЕДІС", "EasyPay"}
EXCLUDE_PARTNERS      = {"Organosyn", "Fishka Online"}

COFFEE_L2_NAME   = "Хот Кафе"

PRE_START    = pd.Timestamp("2025-10-01")
PROMO_START  = pd.Timestamp("2026-01-29")
PROMO_END    = pd.Timestamp("2026-02-28")
SPIKE_START  = pd.Timestamp("2026-03-01")
SPIKE_END    = pd.Timestamp("2026-03-31")
DATA_END     = pd.Timestamp("2026-03-31")
INACTIVITY_CUTOFF = DATA_END - pd.DateOffset(months=2)

# PP score weights (identical to Parts 2-3)
PP_SCORE_WEIGHTS = {
    "retail_monthly_spend_pre": 2.0,
    "pharm_monthly_spend_pre":  1.0,
    "ins_monthly_spend_pre":    1.5,
    "raiff_monthly_spend_pre":  0.5,
    "n_partners_pre":           1.5,
    "pre_median_liters":        1.5,
    "nf_monthly_spend_pre":     1.0,
}
PERIOD_MONTHS = {"pre": 4, "promo": 1, "post_spike": 1}

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def parse_decimal(series):
    return pd.to_numeric(
        series.astype(str)
              .str.replace(",", ".", regex=False)
              .str.replace(r"\s+", "", regex=True),
        errors="coerce"
    )

def tag_period(date):
    if PRE_START <= date < PROMO_START:
        return "pre"
    elif PROMO_START <= date <= PROMO_END:
        return "promo"
    elif SPIKE_START <= date <= SPIKE_END:
        return "post_spike"
    return None

def hr(char="─", w=62):
    print(char * w)

def section(title):
    print()
    hr()
    print(f"  {title}")
    hr()

def field(label, value, indent=2):
    print(f"{'  ' * indent}{label:<40} {value}")


# ─────────────────────────────────────────────────────────────
# DATA LOADING  (global cache — loaded once per session)
# ─────────────────────────────────────────────────────────────
_CACHE = {}

def load_all_data(verbose=False):
    if _CACHE:
        return _CACHE

    def _load_fuel(path):
        df = pd.read_csv(path, sep=SEP)
        df = df.rename(columns={
            "BKT_PRD_CODE": "PRD_CODE",
            "PER_ID_TRN":   "PER_ID",
            "BKT_QUANTITY":  "VOLUME",
            "TRN_PLACE":     "PLC_ID",
        })
        products = pd.read_excel(PRODUCT_FUEL).rename(columns={"PRD_NAME": "FUEL_TYPE"})
        df = df.merge(products, on="PRD_CODE", how="left")
        df["TRN_DATE"] = pd.to_datetime(df["TRN_DATE"], format="%d.%m.%Y")
        df["VOLUME"]   = parse_decimal(df["VOLUME"])
        df["PER_ID"]   = df["PER_ID"].astype(str)
        df["period"]   = df["TRN_DATE"].apply(tag_period)
        return df

    if verbose:
        print("\n  Loading files (this takes ~30s the first time)...")

    fuel_main  = _load_fuel(FUEL_FILE_MAIN)
    fuel_march = _load_fuel(FUEL_FILE_MARCH)

    nf = pd.read_csv(NONFUEL_FILE, sep=SEP)
    nf.columns = nf.columns.str.strip()
    nf["AMOUNT"]   = parse_decimal(nf["AMOUNT"])
    nf["TRN_DATE"] = pd.to_datetime(nf["TRN_DATE"], dayfirst=True)
    nf["PER_ID"]   = nf["PER_ID_TRN"].astype(str)
    nf["period"]   = nf["TRN_DATE"].apply(tag_period)

    par = pd.read_csv(PARTNER_FILE, sep=SEP, encoding=PARTNER_ENCODING)
    par.columns = par.columns.str.strip()
    par["AMOUNT"]   = parse_decimal(par["AMOUNT"])
    par["TRN_DATE"] = pd.to_datetime(par["TRN_DATE"], dayfirst=True)
    par["PER_ID"]   = par["PER_ID"].astype(str)
    par["period"]   = par["TRN_DATE"].apply(tag_period)
    par["partner_type"] = "other"
    par.loc[par["PAR_NAME"].isin(RAIFFEISEN_PARTNERS),    "partner_type"] = "raiffeisen"
    par.loc[par["PAR_NAME"].isin(RETAIL_DISC_PARTNERS),   "partner_type"] = "retail_disc"
    par.loc[par["PAR_NAME"].isin(INSURANCE_MED_PARTNERS), "partner_type"] = "insurance_med"
    par.loc[par["PAR_NAME"].isin(PHARMACY_PARTNERS),      "partner_type"] = "pharmacy"
    par.loc[par["PAR_NAME"].isin(EXCLUDE_PARTNERS),       "partner_type"] = "exclude"
    par_clean = par[par["partner_type"] != "exclude"].copy()

    if verbose:
        print(f"  Fuel (main):  {len(fuel_main):,}  |  Fuel (March): {len(fuel_march):,}")
        print(f"  Non-fuel:     {len(nf):,}  |  Partners:     {len(par_clean):,}")
        print("  Fitting population PP scaler...")

    pop_pp = _fit_population_pp(fuel_main, nf, par_clean)

    _CACHE.update({
        "fuel_main":  fuel_main,
        "fuel_march": fuel_march,
        "nf":         nf,
        "par":        par_clean,
        "pop_pp":     pop_pp,
    })
    if verbose:
        print("  Ready.\n")
    return _CACHE


def _fit_population_pp(fuel_main, nf, par):
    """Fit StandardScaler on full population pre-period components."""
    fuel_pre = fuel_main[fuel_main["period"] == "pre"]
    nf_pre   = nf[nf["period"] == "pre"]
    par_pre  = par[par["period"] == "pre"]

    universe = pd.DataFrame({"PER_ID": fuel_main["PER_ID"].unique()})

    # Fuel median liters (pre)
    vol = fuel_pre.groupby("PER_ID")["VOLUME"].median().reset_index()
    vol.columns = ["PER_ID", "pre_median_liters"]
    universe = universe.merge(vol, on="PER_ID", how="left")

    # Non-fuel monthly spend (pre)
    nf_agg = (nf_pre.groupby("PER_ID")["AMOUNT"].sum() / 4).reset_index()
    nf_agg.columns = ["PER_ID", "nf_monthly_spend_pre"]
    universe = universe.merge(nf_agg, on="PER_ID", how="left")

    # Partner components (pre, monthly)
    for ptype, col in [
        ("retail_disc",   "retail_monthly_spend_pre"),
        ("pharmacy",      "pharm_monthly_spend_pre"),
        ("insurance_med", "ins_monthly_spend_pre"),
        ("raiffeisen",    "raiff_monthly_spend_pre"),
    ]:
        sub = par_pre[par_pre["partner_type"] == ptype]
        agg = (sub.groupby("PER_ID")["AMOUNT"].sum() / 4).reset_index()
        agg.columns = ["PER_ID", col]
        universe = universe.merge(agg, on="PER_ID", how="left")

    # Partner breadth (pre)
    breadth = par_pre.groupby("PER_ID")["PAR_NAME"].nunique().reset_index()
    breadth.columns = ["PER_ID", "n_partners_pre"]
    universe = universe.merge(breadth, on="PER_ID", how="left")

    universe = universe.fillna(0)

    available = [c for c in PP_SCORE_WEIGHTS if c in universe.columns]
    weights   = np.array([PP_SCORE_WEIGHTS[c] for c in available])
    weights   = weights / weights.sum()

    X = universe[available].values
    scaler = StandardScaler()
    scaler.fit(X)

    # Precompute population scores for percentile thresholds
    pop_scores = scaler.transform(X) @ weights
    p33 = float(np.percentile(pop_scores, 33))
    p67 = float(np.percentile(pop_scores, 67))

    return {
        "scaler":    scaler,
        "available": available,
        "weights":   weights,
        "p33":       p33,
        "p67":       p67,
    }


# ─────────────────────────────────────────────────────────────
# CUSTOMER PROFILE
# ─────────────────────────────────────────────────────────────
def compute_profile(per_id, data):
    fuel_main  = data["fuel_main"]
    fuel_march = data["fuel_march"]
    nf         = data["nf"]
    par        = data["par"]
    pop_pp     = data["pop_pp"]

    p = {"per_id": per_id}

    # Existence
    all_ids = set(fuel_main["PER_ID"]) | set(fuel_march["PER_ID"])
    p["found"] = per_id in all_ids
    if not p["found"]:
        return p

    # Slice customer rows
    cf_main  = fuel_main[fuel_main["PER_ID"] == per_id]
    cf_march = fuel_march[fuel_march["PER_ID"] == per_id]
    cf_all   = pd.concat([cf_main, cf_march], ignore_index=True)
    cnf      = nf[nf["PER_ID"] == per_id]
    cpar     = par[par["PER_ID"] == per_id]

    # ── Inactivity ───────────────────────────────────────────
    dates = []
    if len(cf_all) > 0:
        dates.append(cf_all["TRN_DATE"].max())
    if len(cnf) > 0:
        dates.append(cnf["TRN_DATE"].max())
    p["last_date"] = max(dates) if dates else pd.NaT
    p["inactive"]  = pd.isna(p["last_date"]) or p["last_date"] < INACTIVITY_CUTOFF

    # ── Fuel per period ──────────────────────────────────────
    def fuel_stats(df):
        if len(df) == 0:
            return dict(n=0, avg_L=np.nan, median_L=np.nan, total_L=0,
                        p100_n=0, p100_L=0, p100_share=0.0, ever_p100=False)
        p100_mask = df["PRD_CODE"] == PULLS_100_CODE
        return dict(
            n         = len(df),
            avg_L     = df["VOLUME"].mean(),
            median_L  = df["VOLUME"].median(),
            total_L   = df["VOLUME"].sum(),
            p100_n    = int(p100_mask.sum()),
            p100_L    = float(df.loc[p100_mask, "VOLUME"].sum()),
            p100_share= float(p100_mask.mean()),
            ever_p100 = bool(p100_mask.any()),
        )

    p["pre"]  = fuel_stats(cf_main[cf_main["period"] == "pre"])
    p["promo"]= fuel_stats(cf_main[cf_main["period"] == "promo"])
    p["post"] = fuel_stats(cf_march[cf_march["period"] == "post_spike"])

    # All-time avg/median (most stable signal for intervention thresholds)
    p["avg_L_alltime"]    = cf_all["VOLUME"].mean() if len(cf_all) > 0 else np.nan
    p["median_L_alltime"] = cf_all["VOLUME"].median() if len(cf_all) > 0 else np.nan

    # Fuel type counts (verbose)
    if len(cf_all) > 0:
        p["fuel_types"] = (cf_all.groupby("FUEL_TYPE")["VOLUME"]
                           .agg(n="count", liters="sum")
                           .sort_values("n", ascending=False))
    else:
        p["fuel_types"] = pd.DataFrame()

    is_p100 = p["pre"]["ever_p100"] or p["post"]["ever_p100"]
    p["is_p100_buyer"] = is_p100

    # ── P100 engagement status ───────────────────────────────
    # Active:  bought P100 in the last 2 months (Feb promo or March)
    # Lapsed:  bought P100 before but not in the last 2 months
    # Never:   no P100 transactions at all
    last_p100_date = (
        cf_all[cf_all["PRD_CODE"] == PULLS_100_CODE]["TRN_DATE"].max()
        if is_p100 else pd.NaT
    )
    p["last_p100_date"] = last_p100_date
    if not is_p100:
        p["p100_status"] = "never"
    elif pd.notna(last_p100_date) and last_p100_date >= INACTIVITY_CUTOFF:
        p["p100_status"] = "active"
    else:
        p["p100_status"] = "lapsed"

    # ── P100 share tier (across full history) ────────────────
    # Uses all-time share for the most stable signal.
    # Tiers: <20% / 20-50% / 50-70% / 70%+
    alltime_p100_share = (
        float((cf_all["PRD_CODE"] == PULLS_100_CODE).mean())
        if len(cf_all) > 0 else 0.0
    )
    p["p100_share_alltime"] = alltime_p100_share
    if not is_p100 or alltime_p100_share == 0:
        p["p100_share_tier"] = "none"
    elif alltime_p100_share < 0.20:
        p["p100_share_tier"] = "<20%"
    elif alltime_p100_share < 0.50:
        p["p100_share_tier"] = "20-50%"
    elif alltime_p100_share < 0.70:
        p["p100_share_tier"] = "50-70%"
    else:
        p["p100_share_tier"] = "70%+"

    # ── Non-fuel per period ──────────────────────────────────
    def nf_stats(df, months=1):
        if len(df) == 0:
            return dict(n=0, monthly_spend=0, n_cats=0,
                        has_coffee=False, coffee_spend=0)
        has_coffee = False
        coffee_spend = 0
        if "L2_TRR_TN_NAME" in df.columns:
            coffee_rows = df[df["L2_TRR_TN_NAME"].astype(str).str.contains(
                COFFEE_L2_NAME, na=False, case=False)]
            has_coffee   = len(coffee_rows) > 0
            coffee_spend = float(coffee_rows["AMOUNT"].sum())
        n_cats = df["L3_TRR_TN_USER_CODE"].nunique() if "L3_TRR_TN_USER_CODE" in df.columns else 0
        return dict(
            n             = len(df),
            monthly_spend = float(df["AMOUNT"].sum()) / months,
            n_cats        = n_cats,
            has_coffee    = has_coffee,
            coffee_spend  = coffee_spend,
        )

    p["nf_pre"]  = nf_stats(cnf[cnf["period"] == "pre"],  months=4)
    p["nf_post"] = nf_stats(cnf[cnf["period"] == "post_spike"], months=1)

    p["buys_nonfuel"] = p["nf_pre"]["n"] >= 2  # ≥2 pre-period trns to avoid single-visit noise
    p["buys_coffee"]  = p["nf_pre"]["has_coffee"] or p["nf_post"]["has_coffee"]

    pre_nf_monthly  = p["nf_pre"]["monthly_spend"]
    post_nf_monthly = p["nf_post"]["monthly_spend"]
    p["nf_change_pct"] = (
        (post_nf_monthly - pre_nf_monthly) / pre_nf_monthly * 100
        if pre_nf_monthly > 0 else 0.0
    )
    p["nf_dropped"] = p["nf_change_pct"] < -20

    # ── Partner signals ───────────────────────────────────────
    par_pre = cpar[cpar["period"] == "pre"]
    p["partner_names"]  = sorted(cpar["PAR_NAME"].unique().tolist())
    p["n_partners_pre"] = int(par_pre["PAR_NAME"].nunique())

    # Pharmacy-only flag: all pre-period partner spend is pharmacy
    if len(par_pre) > 0:
        p["pharmacy_only"] = set(par_pre["partner_type"].unique()) <= {"pharmacy"}
    else:
        p["pharmacy_only"] = False

    for ptype, key in [
        ("retail_disc",   "par_retail_pre"),
        ("pharmacy",      "par_pharm_pre"),
        ("insurance_med", "par_ins_pre"),
        ("raiffeisen",    "par_raiff_pre"),
    ]:
        sub = par_pre[par_pre["partner_type"] == ptype]
        p[key] = float(sub["AMOUNT"].sum()) / 4 if len(sub) > 0 else 0.0

    # ── PP Score ─────────────────────────────────────────────
    available = pop_pp["available"]
    scaler    = pop_pp["scaler"]
    weights   = pop_pp["weights"]

    fvec = {
        "retail_monthly_spend_pre": p["par_retail_pre"],
        "pharm_monthly_spend_pre":  p["par_pharm_pre"],
        "ins_monthly_spend_pre":    p["par_ins_pre"],
        "raiff_monthly_spend_pre":  p["par_raiff_pre"],
        "n_partners_pre":           float(p["n_partners_pre"]),
        "pre_median_liters":        p["pre"]["median_L"] if not np.isnan(p["pre"]["median_L"]) else 0.0,
        "nf_monthly_spend_pre":     p["nf_pre"]["monthly_spend"],
    }
    X = np.array([[fvec.get(c, 0) for c in available]])
    X_scaled = scaler.transform(X)
    p["pp_score"]   = float((X_scaled * weights).sum())
    p["pp_tercile"] = (
        "high" if p["pp_score"] >= pop_pp["p67"] else
        "mid"  if p["pp_score"] >= pop_pp["p33"] else
        "low"
    )

    return p


# ─────────────────────────────────────────────────────────────
# INTERVENTION LOGIC  (v2 — full multi-factor decision tree)
# ─────────────────────────────────────────────────────────────
def classify(p):
    """
    Returns (code, reason, [flags]).

    Decision priority:
        1. Universal exclusions  → --
        2. ACTIVE P100           → A1 / A2
        3. LAPSED P100           → A1 / A2 / --
        4. NEVER P100            → A1 / A2 / B1 / --

    Within each block, volume flag (≥30L) takes precedence over PP flag.
    Coffee-buyer sub-flag enriches A1 reason text wherever A1 is assigned.
    """
    flags = []

    # ── Universal exclusions ─────────────────────────────────
    if not p["found"]:
        return "--", "Customer ID not found in transaction data.", flags

    if p["inactive"]:
        ds = p["last_date"].strftime("%d.%m.%Y") if pd.notna(p["last_date"]) else "unknown"
        return "--", f"No transactions since {ds} — inactive for more than 2 months.", flags

    if p["pharmacy_only"]:
        flags.append("pharmacy_only=True — partner spend is necessity-only, not a PP signal")
        return "--", (
            "Partner transactions are pharmacy-only (necessity spending). "
            "PP signal is not reliable; no intervention recommended."
        ), flags

    # ── Convenience variables ────────────────────────────────
    p100_status = p["p100_status"]      # "active" | "lapsed" | "never"
    share_tier  = p["p100_share_tier"]  # "none" | "<20%" | "20-50%" | "50-70%" | "70%+"
    buys_nf     = p["buys_nonfuel"]     # ≥2 pre-period non-fuel transactions
    buys_coffee = p["buys_coffee"]
    nf_dropped  = p["nf_dropped"]       # non-fuel spend dropped >20% in March
    pp_high     = p["pp_tercile"] == "high"

    # Best avg-liters signal: prefer pre-period; fall back to all-time
    avg_L = (
        p["pre"]["avg_L"]
        if (p["pre"]["n"] > 0 and not np.isnan(p["pre"]["avg_L"]))
        else p["avg_L_alltime"]
        if pd.notna(p["avg_L_alltime"])
        else 0.0
    )
    vol_high = avg_L >= 30  # volume threshold flag

    # Coffee bundle note — appended to A1 reason wherever applicable
    def coffee_note():
        return " Coffee buyer → bundle offer targets an existing station habit." if buys_coffee else ""

    last_p100_ds = (
        p["last_p100_date"].strftime("%d.%m.%Y")
        if pd.notna(p.get("last_p100_date")) else "unknown"
    )

    # ════════════════════════════════════════════════════════
    # BLOCK 1 — ACTIVE P100
    # ════════════════════════════════════════════════════════
    if p100_status == "active":

        # A1-Act-1: non-fuel spend dropped >20% in March (Shapiro spike signal)
        # Bundle re-engages station habits hurt by the price shock
        if buys_nf and nf_dropped:
            flags.append(
                f"active P100 | nf_dropped={p['nf_change_pct']:.0f}% | share={share_tier}"
            )
            return "A1", (
                f"Active P100 buyer (share: {share_tier}): non-fuel spend dropped "
                f"{p['nf_change_pct']:.0f}% in March after the price spike. "
                f"Coffee bundle re-anchors station visit value.{coffee_note()}"
            ), flags

        # A1-Act-2: partial P100 (<20% share) + has non-fuel
        # They use P100 occasionally; bundle makes it feel like the 'station visit' choice
        if share_tier == "<20%" and buys_nf:
            flags.append(
                f"active P100 | share=<20% | buys_nonfuel=True | avg_L={avg_L:.1f}"
            )
            return "A1", (
                f"Active but infrequent P100 buyer (<20% share, {avg_L:.1f}L avg). "
                f"Has station non-fuel habits — bundle deepens P100 association with visits.{coffee_note()}"
            ), flags

        # A1-Act-3: moderate share (20–50%) + has non-fuel
        # Growing share + station engagement → reinforce with bundle
        if share_tier == "20-50%" and buys_nf:
            flags.append(
                f"active P100 | share=20-50% | buys_nonfuel=True | avg_L={avg_L:.1f}"
            )
            return "A1", (
                f"Active P100 buyer (20–50% share, {avg_L:.1f}L avg) with station non-fuel habits. "
                f"Bundle reinforces the P100-station pairing and nudges share upward.{coffee_note()}"
            ), flags

        # A2-Act-1: avg ≥30L (any share tier)
        # High-volume active buyer → discount rewards the habit
        if vol_high:
            if share_tier == "70%+":
                detail = "Near-exclusive P100 buyer — discount is a loyalty reward."
            elif share_tier in ("50-70%", "20-50%"):
                detail = f"Mixed buyer ({share_tier} share) — discount to push toward full preference."
            else:
                detail = f"Occasional P100 buyer ({share_tier} share) — discount to deepen habit."
            flags.append(
                f"active P100 | avg_L={avg_L:.1f} (≥30) | share={share_tier}"
            )
            return "A2", (
                f"Active P100 buyer averaging {avg_L:.1f}L/fill. {detail}"
            ), flags

        # A2-Act-2: high share (50–70% or 70%+), any volume — reward loyalty
        if share_tier in ("50-70%", "70%+"):
            detail = (
                "Near-exclusive buyer — discount as loyalty recognition."
                if share_tier == "70%+"
                else "Committed buyer — discount to push toward exclusivity."
            )
            flags.append(
                f"active P100 | share={share_tier} | avg_L={avg_L:.1f}"
            )
            return "A2", (
                f"Active P100 buyer ({share_tier} share, {avg_L:.1f}L avg). {detail}"
            ), flags

        # Active + low volume + low/mid PP + no non-fuel → nothing strong enough
        flags.append(
            f"active P100 | share={share_tier} | avg_L={avg_L:.1f} | pp={p['pp_tercile']} "
            f"| buys_nonfuel={buys_nf} — no strong lever"
        )
        return "--", (
            f"Active P100 buyer but low volume ({avg_L:.1f}L avg), low-to-mid PP, "
            f"and no station non-fuel habits. No high-confidence intervention."
        ), flags

    # ════════════════════════════════════════════════════════
    # BLOCK 2 — LAPSED P100
    # ════════════════════════════════════════════════════════
    if p100_status == "lapsed":

        # A1-Lap-1: ≥30L + has non-fuel
        # Bundle is a soft re-entry hook; label notes this is a lapsed buyer
        if vol_high and buys_nf:
            flags.append(
                f"lapsed P100 | last={last_p100_ds} | avg_L={avg_L:.1f} "
                f"| buys_nonfuel=True | share={share_tier}"
            )
            return "A1", (
                f"Lapsed P100 buyer (last purchase: {last_p100_ds}, historical share: {share_tier}). "
                f"Averaging {avg_L:.1f}L/fill with station non-fuel habits. "
                f"Coffee bundle is a softer re-entry hook than a pure discount.{coffee_note()}"
            ), flags

        # A1-Lap-2: non-fuel dropped >20% + has non-fuel (even if <30L)
        # Same Shapiro spike logic — the price shock may have driven them off
        if buys_nf and nf_dropped:
            flags.append(
                f"lapsed P100 | nf_dropped={p['nf_change_pct']:.0f}% | avg_L={avg_L:.1f}"
            )
            return "A1", (
                f"Lapsed P100 buyer whose non-fuel spend dropped {p['nf_change_pct']:.0f}% "
                f"in March. Station habits weakened by the price spike — "
                f"coffee bundle may restore both.{coffee_note()}"
            ), flags

        # A1-Lap-3: high PP + has non-fuel (replaces broad "any nonfuel" catch-all)
        # Means are there AND station habit exists — bundle is well-targeted
        if pp_high and buys_nf:
            flags.append(
                f"lapsed P100 | pp=high | avg_L={avg_L:.1f} | buys_nonfuel=True | share={share_tier}"
            )
            return "A1", (
                f"Lapsed P100 buyer (last: {last_p100_ds}, share: {share_tier}), {avg_L:.1f}L avg. "
                f"High PP with station non-fuel habits — coffee bundle as targeted re-entry hook.{coffee_note()}"
            ), flags

        # A2-Lap-1: ≥30L + no non-fuel
        # No bundling hook → pure price discount is the right lever
        if vol_high and not buys_nf:
            flags.append(
                f"lapsed P100 | avg_L={avg_L:.1f} (≥30) | buys_nonfuel=False | share={share_tier}"
            )
            return "A2", (
                f"Lapsed P100 buyer (last: {last_p100_ds}, share: {share_tier}), {avg_L:.1f}L avg. "
                f"No station non-fuel habits — price discount is the cleaner re-entry lever."
            ), flags

        # B1-Lap: high PP + no non-fuel (any volume)
        # Has means but no station habit and no volume hook — educate on P100/E0 value
        # before deploying a discount that may not stick
        if pp_high and not buys_nf:
            flags.append(
                f"lapsed P100 | pp=high | avg_L={avg_L:.1f} | buys_nonfuel=False"
            )
            return "B1", (
                f"Lapsed P100 buyer (last: {last_p100_ds}, share: {share_tier}) with high PP "
                f"but no non-fuel station habits. Educate on P100/E0 benefits — "
                f"no bundle hook available and a discount alone lacks durability."
            ), flags

        # Low PP + low volume + no non-fuel → no strong lever
        flags.append(
            f"lapsed P100 | avg_L={avg_L:.1f} | pp={p['pp_tercile']} "
            f"| buys_nonfuel={buys_nf} — low PP, low vol, no station habit"
        )
        return "--", (
            f"Lapsed P100 buyer (last: {last_p100_ds}) but low volume ({avg_L:.1f}L), "
            f"low PP, and no station non-fuel habits. No reliable intervention lever."
        ), flags

    # ════════════════════════════════════════════════════════
    # BLOCK 3 — NEVER P100
    # ════════════════════════════════════════════════════════
    # p100_status == "never" implied from here

    # A1-Nev-1: ≥30L + has non-fuel
    # High-volume station-dweller who never tried P100 → bundle is the entry point
    if vol_high and buys_nf:
        flags.append(
            f"never P100 | avg_L={avg_L:.1f} (≥30) | buys_nonfuel=True | pp={p['pp_tercile']}"
        )
        return "A1", (
            f"Never bought P100, averaging {avg_L:.1f}L/fill with station non-fuel habits. "
            f"Bundle P100 trial with a coffee promo — station engagement is the hook.{coffee_note()}"
        ), flags

    # A2-Nev-1: high PP + any volume (no non-fuel)
    # Means available; barrier is awareness/price, not budget
    if pp_high:
        flags.append(
            f"never P100 | pp=high | avg_L={avg_L:.1f} | buys_nonfuel={buys_nf}"
        )
        return "A2", (
            f"High PP (tercile: high, score: {p['pp_score']:.3f}) but never bought P100. "
            f"Averaging {avg_L:.1f}L/fill. Discount lowers the first-adoption barrier."
        ), flags

    # A2-Nev-2: ≥30L + no non-fuel + any PP
    # High-volume fueler who never engaged with P100 or station services
    # Discount is the available lever; education alone (B1) loses to a concrete offer
    # BUT only if mid PP — low PP + no non-fuel → B1 or nothing
    if vol_high and not buys_nf and p["pp_tercile"] == "mid":
        flags.append(
            f"never P100 | avg_L={avg_L:.1f} (≥30) | buys_nonfuel=False | pp=mid"
        )
        return "A2", (
            f"Never bought P100, {avg_L:.1f}L avg, mid PP, no station non-fuel purchases. "
            f"Volume signals willingness — discount to prompt first trial."
        ), flags

    # B1: ≥30L + no non-fuel + low PP
    # They have the fill behavior but no station engagement and constrained budget —
    # educate first; a discount risks margin without durability
    if vol_high and not buys_nf:
        flags.append(
            f"never P100 | avg_L={avg_L:.1f} (≥30) | buys_nonfuel=False | pp={p['pp_tercile']}"
        )
        return "B1", (
            f"Never bought P100, averaging {avg_L:.1f}L/fill, low PP, no station non-fuel purchases. "
            f"Educate on P100 / E0 benefits — no bundling hook and budget may constrain adoption."
        ), flags

    # Never + <30L + low-mid PP + no non-fuel → nothing
    flags.append(
        f"never P100 | avg_L={avg_L:.1f} (<30) | pp={p['pp_tercile']} "
        f"| buys_nonfuel={buys_nf} — no qualifying criteria"
    )
    return "--", (
        f"Never bought P100. Low volume ({avg_L:.1f}L avg), low-to-mid PP, "
        f"no station non-fuel habits. No high-confidence intervention."
    ), flags


# ─────────────────────────────────────────────────────────────
# DISPLAY
# ─────────────────────────────────────────────────────────────
LABELS = {
    "A1": "A1 — BUNDLING OFFER        (coffee promo)",
    "A2": "A2 — DISCOUNT COUPON       (-3 UAH/L on P100)",
    "B1": "B1 — EDUCATIONAL PUSH      (P100 / E0 awareness)",
    "--": "DO NOT RECOMMEND",
}

def display(p, code, reason, flags, verbose=False):
    pid = p["per_id"]

    section(f"CUSTOMER  {pid}")

    if not p["found"]:
        print(f"\n  Customer {pid} not found in any transaction file.\n")
        _print_verdict(code, reason)
        return

    # Activity
    ds = p["last_date"].strftime("%d.%m.%Y") if pd.notna(p["last_date"]) else "unknown"
    status = "⚠  INACTIVE >2 months" if p["inactive"] else "active"
    field("Last transaction:", f"{ds}  [{status}]")

    # ── Fuel ─────────────────────────────────────────────────
    section("FUEL BEHAVIOR")
    pre  = p["pre"]
    post = p["post"]

    def lstr(v):
        return f"{v:.1f}L" if pd.notna(v) and not np.isnan(v) else "n/a"

    field("Pre-period fills (Oct–Jan):", str(pre["n"]))
    field("Avg liters/fill (pre):", lstr(pre["avg_L"]))
    field("Median liters/fill (pre):", lstr(pre["median_L"]))
    field("Avg liters/fill (all-time):", lstr(p["avg_L_alltime"]))

    # P100 status block
    status_display = {
        "active": "Active ✓  (purchased within last 2 months)",
        "lapsed": "Lapsed ⚠  (no P100 in last 2+ months)",
        "never":  "Never bought P100",
    }
    field("P100 status:", status_display.get(p["p100_status"], p["p100_status"]))
    if p["is_p100_buyer"]:
        last_ds = p["last_p100_date"].strftime("%d.%m.%Y") if pd.notna(p["last_p100_date"]) else "unknown"
        field("  Last P100 purchase:", last_ds)
        field("  P100 share (all-time):",
              f"{p['p100_share_alltime']*100:.1f}%  [{p['p100_share_tier']}]")
        field("  P100 transactions (pre):", str(pre["p100_n"]))
        field("  P100 liters (pre):", f"{pre['p100_L']:.1f}L")
        field("  P100 share of fills (pre):", f"{pre['p100_share']*100:.1f}%")
    if post["n"] > 0:
        field("March fills:", str(post["n"]))
        field("  Avg liters (March):", lstr(post["avg_L"]))
        field("  Bought P100 in March:", "Yes ✓" if post["ever_p100"] else "No")
    else:
        field("March transactions:", "None")

    if verbose and len(p["fuel_types"]) > 0:
        print("\n  Fuel type breakdown (all history):")
        for ftype, row in p["fuel_types"].iterrows():
            print(f"    {str(ftype):<24} {int(row['n']):>4} fills   "
                  f"{row['liters']:.1f}L total")

    # ── Non-fuel ─────────────────────────────────────────────
    section("NON-FUEL (OKKO station)")
    nfp = p["nf_pre"]
    nfpost = p["nf_post"]
    field("Buys non-fuel:", "Yes ✓" if p["buys_nonfuel"] else "No")
    if p["buys_nonfuel"]:
        field("  Monthly spend (pre):", f"{nfp['monthly_spend']:.0f} UAH")
        field("  Unique L3 categories (pre):", str(nfp["n_cats"]))
        field("  Buys coffee:", "Yes ✓" if p["buys_coffee"] else "No")
        if nfpost["n"] > 0:
            field("  Monthly spend (March):", f"{nfpost['monthly_spend']:.0f} UAH")
            chg = p["nf_change_pct"]
            drop_note = "  ⚠ large drop" if p["nf_dropped"] else ""
            field("  Change vs pre:", f"{chg:+.1f}%{drop_note}")

    # ── Partners ─────────────────────────────────────────────
    section("PARTNER TRANSACTIONS (Fishka network)")
    if p["partner_names"]:
        field("Partners:", ", ".join(p["partner_names"]))
        field("Pharmacy-only flag:",
              "⚠  Yes (necessity-driven)" if p["pharmacy_only"] else "No")
        if verbose:
            field("  Retail disc. monthly (pre):", f"{p['par_retail_pre']:.0f} UAH")
            field("  Pharmacy monthly (pre):",     f"{p['par_pharm_pre']:.0f} UAH")
            field("  Raiffeisen monthly (pre):",   f"{p['par_raiff_pre']:.0f} UAH")
            field("  Insurance/med monthly (pre):",f"{p['par_ins_pre']:.0f} UAH")
    else:
        field("Partner transactions:", "None in dataset")

    # ── PP Score ─────────────────────────────────────────────
    section("PURCHASING POWER")
    field("PP Score:", f"{p['pp_score']:.4f}")
    field("PP Tercile:", p["pp_tercile"].upper())
    p33, p67 = p.get("pp_thresholds", (None, None))

    # ── Flags ─────────────────────────────────────────────────
    if flags:
        section("DECISION FLAGS")
        for f in flags:
            print(f"  •  {f}")

    _print_verdict(code, reason)


def _print_verdict(code, reason):
    print()
    hr("═")
    label = LABELS.get(code, code)
    print(f"  VERDICT:  {label}")
    print(f"  REASON:   {reason}")
    hr("═")
    print()


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="OKKO Customer Intervention Classifier")
    parser.add_argument("--id",      type=str,  default=None,
                        help="Customer PER_ID. Omit for interactive mode.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show fuel type breakdown and per-category partner spend.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force reload all data files.")
    args = parser.parse_args()

    if args.no_cache:
        _CACHE.clear()

    # File existence check
    required = [FUEL_FILE_MAIN, FUEL_FILE_MARCH, NONFUEL_FILE,
                PARTNER_FILE, PRODUCT_FUEL]
    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        print("\n⚠  Missing data files:")
        for f in missing:
            print(f"   {f}")
        print("\nCheck the CONFIG paths at the top of the script.")
        sys.exit(1)

    print("\n" + "═" * 62)
    print("  OKKO CUSTOMER INTERVENTION CLASSIFIER  v3")
    print("═" * 62)

    data = load_all_data(verbose=True)

    while True:
        if args.id:
            per_id = args.id.strip()
        else:
            try:
                per_id = input("  Enter PER_ID  (or 'q' to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  Exiting.\n")
                break

        if per_id.lower() in ("q", "quit", "exit", ""):
            print("\n  Exiting.\n")
            break

        profile = compute_profile(per_id, data)
        code, reason, flags = classify(profile)
        display(profile, code, reason, flags, verbose=args.verbose)

        if args.id:
            break


if __name__ == "__main__":
    main()
