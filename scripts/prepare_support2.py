"""Build a leakage-free modelling table from the SUPPORT2 study.

SUPPORT2 (Study to Understand Prognoses and Preferences for Outcomes and Risks
of Treatments) follows 9,105 seriously-ill hospitalised patients. Source:
https://hbiostat.org/data/repo/support2csv.zip -- public, no authentication.

The point of this script is the *exclusions*. SUPPORT2 ships several columns
that would leak the outcome if handed to a survival model, and several more
that are measured after baseline. Both kinds are dropped here, with the reason
recorded next to each one, so the modelling table can be trusted downstream.

Writes data/support2_modelling.csv.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://hbiostat.org/data/repo/support2csv.zip"
ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "support2.csv"
OUT = ROOT / "data" / "support2_modelling.csv"

# ---------------------------------------------------------------------------
# What we deliberately do NOT model, and why.
# ---------------------------------------------------------------------------
EXCLUDED = {
    "surv2m": "the SUPPORT model's own predicted 2-month survival -- leakage",
    "surv6m": "the SUPPORT model's own predicted 6-month survival -- leakage",
    "prg2m": "physician's subjective survival estimate -- a prediction, not a covariate",
    "prg6m": "physician's subjective survival estimate -- a prediction, not a covariate",
    "hospdead": "outcome (died in hospital), not baseline",
    "slos": "length of stay -- accrues after baseline",
    "charges": "billed charges -- accrue over the stay",
    "totcst": "total cost -- accrues over the stay",
    "totmcst": "total micro-cost -- accrues over the stay",
    "sfdm2": "functional disability measured at 2 months -- post-baseline",
    "dnr": "do-not-resuscitate order -- can be written after baseline",
    "dnrday": "day the DNR order was written -- post-baseline",
    "aps": "APACHE III physiology score -- a composite of the vitals/labs already modelled",
    "sps": "SUPPORT physiology score -- a composite of the vitals/labs already modelled",
    "adlp": "superseded by adlsc (adlp is 62% missing)",
    "adls": "superseded by adlsc (adls is 31% missing)",
    "glucose": "49% missing",
    "bun": "48% missing",
    "urine": "53% missing",
    "alb": "37% missing",
    "pafi": "26% missing",
    "ph": "25% missing",
    "bili": "29% missing",
    "edu": "18% missing",
    "income": "33% missing",
    "hday": "day of study entry relative to admission -- design artefact",
    "dzclass": "coarser recoding of dzgroup; keeping both would double-count",
}

CONTINUOUS = [
    "age", "meanbp", "hrt", "resp", "temp",   # demographics + vitals
    "wblc", "sod", "crea",                     # labs
    "num.co", "scoma", "adlsc",                # burden / function
]
BINARY_RAW = ["diabetes", "dementia"]

# Seven levels, not eight: "ARF/MOSF w/Sepsis" is the omitted reference. Reference
# levels are the largest category throughout, so the intercept describes the modal
# patient rather than a rare one.
DZ_LEVELS = ["CHF", "COPD", "Lung Cancer", "MOSF w/Malig", "Coma", "Colon Cancer", "Cirrhosis"]
DZ_SLUG = {
    "CHF": "dz_chf", "COPD": "dz_copd", "Lung Cancer": "dz_lung_ca",
    "MOSF w/Malig": "dz_mosf_malig", "Coma": "dz_coma",
    "Colon Cancer": "dz_colon_ca", "Cirrhosis": "dz_cirrhosis",
}
RACE_LEVELS = ["black", "hispanic", "other", "asian"]   # reference: white

# Clinical grouping of the covariates. Used to document what is in the table and
# to define BENCH14 below by subtraction.
BLOCKS: dict[str, list[str]] = {
    "vitals": ["meanbp", "hrt", "resp", "temp"],
    "labs": ["wblc", "sod", "crea"],
    "comorbid": ["num.co", "diabetes", "dementia", "ca_metastatic", "ca_yes"],
    "function": ["scoma", "adlsc"],
    "demog": ["age", "sex_female", "race_black", "race_hispanic", "race_other", "race_asian"],
    "disease": list(DZ_SLUG.values()),
}
FEATURES = [c for cols in BLOCKS.values() for c in cols]

# ---------------------------------------------------------------------------
# The published-benchmark covariate set.
#
# Kvamme, Borgan & Scheel (2019), JMLR 20(129) and Katzman et al. (2018) both
# model SUPPORT with 14 conceptual covariates: age, sex, race, num.co, diabetes,
# dementia, ca, meanbp, hrt, resp, temp, wblc, sod, crea. Encoded (race -> 4
# dummies, ca -> 2) that is 18 columns. Notably it excludes disease group, which
# is why our C-index drops from ~0.68 to ~0.60 on this set -- the price of being
# directly comparable to published numbers.
# ---------------------------------------------------------------------------
BENCH14 = [f for f in FEATURES
           if f not in BLOCKS["disease"] and f not in BLOCKS["function"]]



# ---------------------------------------------------------------------------
# Train/test fold. Kept here so every consumer scores on byte-identical data.
# ---------------------------------------------------------------------------
TEST_FRAC = 0.30
SEED = 20260819

# Indicators are already on a sensible scale; only continuous covariates are
# standardised, and always on TRAIN moments so nothing leaks from the test fold.
BINARY = {
    "diabetes", "dementia", "sex_female", "race_black", "race_hispanic",
    "race_other", "race_asian", "ca_metastatic", "ca_yes",
    *BLOCKS["disease"],
}


def stratified_split(d: pd.DataFrame, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Split within each dzgroup x event cell so both folds match on case mix."""
    test_idx: list[int] = []
    for _, cell in d.groupby(["dzgroup", "event"], observed=True):
        idx = cell.index.to_numpy()
        rng.shuffle(idx)
        test_idx.extend(idx[: int(round(TEST_FRAC * len(idx)))])
    test_mask = np.zeros(len(d), dtype=bool)
    test_mask[np.array(sorted(test_idx))] = True
    return ~test_mask, test_mask


def prepare_fold(features: list[str] | None = None):
    """The single source of truth for the SUPPORT2 train/test fold.

    Returns ``(X_train, X_test, t_train_days, t_test_days, event_train, event_test)``.
    Continuous covariates are standardised on TRAIN moments only; times are in days.
    """
    features = list(FEATURES) if features is None else list(features)
    cont = [c for c in features if c not in BINARY]

    d = pd.read_csv(OUT)
    rng = np.random.default_rng(SEED)
    tr_mask, te_mask = stratified_split(d, rng)
    train, test = d[tr_mask].reset_index(drop=True), d[te_mask].reset_index(drop=True)

    mu_, sd_ = train[cont].mean(), train[cont].std()
    Xtr, Xte = train[features].copy(), test[features].copy()
    Xtr[cont] = (Xtr[cont] - mu_) / sd_
    Xte[cont] = (Xte[cont] - mu_) / sd_

    return (Xtr, Xte,
            train.t_days.to_numpy(float), test.t_days.to_numpy(float),
            train.event.to_numpy(int), test.event.to_numpy(int))


def download() -> pd.DataFrame:
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL} ...")
        with urllib.request.urlopen(URL) as r:
            blob = r.read()
        with zipfile.ZipFile(io.BytesIO(blob)) as z:
            name = next(n for n in z.namelist() if n.endswith(".csv"))
            RAW.write_bytes(z.read(name))
        print(f"  wrote {RAW} ({RAW.stat().st_size / 1e6:.1f} MB)")
    return pd.read_csv(RAW)


def main() -> None:
    d = download()
    print(f"\nraw SUPPORT2: {d.shape[0]:,} rows x {d.shape[1]} columns")

    print(f"\ndropping {len(EXCLUDED)} columns:")
    for col, why in EXCLUDED.items():
        if col in d.columns:
            print(f"  {col:10} {why}")

    out = pd.DataFrame(index=d.index)

    # ---- outcome -----------------------------------------------------------
    # d.time is follow-up in days (3 .. 2029); death is the event indicator.
    out["t_days"] = d["d.time"].astype(float)
    out["event"] = d["death"].astype(int)

    # ---- covariates --------------------------------------------------------
    for c in CONTINUOUS:
        out[c] = pd.to_numeric(d[c], errors="coerce")
    for c in BINARY_RAW:
        out[c] = d[c].astype(float)

    out["sex_female"] = (d["sex"] == "female").astype(float)
    for lvl in RACE_LEVELS:
        out[f"race_{lvl}"] = (d["race"] == lvl).astype(float)
    # race is NaN for 42 rows; those must not silently become "white"
    out.loc[d["race"].isna(), [f"race_{lvl}" for lvl in RACE_LEVELS]] = np.nan

    out["ca_metastatic"] = (d["ca"] == "metastatic").astype(float)
    out["ca_yes"] = (d["ca"] == "yes").astype(float)

    for lvl in DZ_LEVELS:
        out[DZ_SLUG[lvl]] = (d["dzgroup"] == lvl).astype(float)
    out["dzgroup"] = d["dzgroup"]          # kept for stratifying the split

    # ---- complete case -----------------------------------------------------
    before = len(out)
    out = out.dropna(subset=FEATURES + ["t_days", "event"]).reset_index(drop=True)
    out = out[out["t_days"] > 0].reset_index(drop=True)
    print(f"\ncomplete case: {len(out):,} of {before:,} rows "
          f"({100 * len(out) / before:.1f}%)")

    assert list(out.columns) == ["t_days", "event"] + CONTINUOUS + BINARY_RAW + [
        "sex_female", *(f"race_{lvl}" for lvl in RACE_LEVELS), "ca_metastatic", "ca_yes",
        *DZ_SLUG.values(), "dzgroup",
    ], "column layout drifted"
    assert set(FEATURES) <= set(out.columns)
    assert len(FEATURES) == 27, f"expected 27 covariates, built {len(FEATURES)}"
    assert len(BENCH14) == 18, f"expected 18 benchmark covariates, built {len(BENCH14)}"

    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)

    print(f"\ncovariates: {len(FEATURES)} in {len(BLOCKS)} blocks")
    for name, cols in BLOCKS.items():
        print(f"  {name:9} ({len(cols)}) {', '.join(cols)}")
    print(f"\nevent rate      {100 * out.event.mean():.1f}%")
    print(f"follow-up days  min {out.t_days.min():.0f}  median {out.t_days.median():.0f}  "
          f"max {out.t_days.max():.0f}")
    print(f"median time among events: {out.loc[out.event == 1, 't_days'].median():.0f} days")
    print(f"\nwrote {OUT}  ({out.shape[0]:,} x {out.shape[1]})")


if __name__ == "__main__":
    main()
