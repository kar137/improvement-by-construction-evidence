#!/usr/bin/env python
"""
sambodhan_repro.py -- shared, deterministic reimplementation of the frozen
repository's own data preparation, plus the measurement primitives used by
every other script in paper1-evidence/scripts/.

Nothing here reads or writes inside the Sambodhan repository except to READ
its CSV files. The repository is treated as immutable evidence.

The split functions are line-for-line reimplementations of:
  * src/data_science/preprocessing/data_prep.py:10-34            (urgency)
  * src/data_science/models/dept_classifier/
        preprocess_and_prepare_dataset.py:31-103                 (department)
They are reimplemented rather than imported because the department module
imports `datasets`, which is not installed locally.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
EVIDENCE = Path(__file__).resolve().parent.parent
REPO = EVIDENCE.parent / "Sambodhan-AI-Powered-Grievance-Redressal-System-for-Local-Governance"

CORPUS_6000 = REPO / "data/processed/sambodhan_balanced_dataset.csv"
CORPUS_1640 = REPO / "data/processed/final-grievance-data_with_urgency-dept.csv"
RAW_DETAILS = REPO / "data/raw/sambodhan_complaint_details.csv"
RAW_TWEETS = REPO / "data/raw/csv/Grievance_Tweets_India_RailMin_IncomeTax_DelhiPolice.csv"
RAW_CATEG = REPO / "data/raw/csv/categorized_grievances_dataset.csv"

# --------------------------------------------------------------------------
# Label maps -- copied verbatim from the repository
# --------------------------------------------------------------------------
URGENCY2ID = {"NORMAL": 0, "URGENT": 1, "HIGHLY URGENT": 2}
ID2URGENCY = {v: k for k, v in URGENCY2ID.items()}
DEPARTMENT2ID = {
    "Municipal Governance & Community Services": 0,
    "Education, Health & Social Welfare": 1,
    "Infrastructure, Utilities & Natural Resources": 2,
    "Security & Law Enforcement": 3,
}
ID2DEPARTMENT = {v: k for k, v in DEPARTMENT2ID.items()}

GOLD_URGENCY_VALUES = ["NORMAL", "URGENT", "HIGHLY URGENT"]


# --------------------------------------------------------------------------
# Text cleaning -- verbatim reimplementations
# --------------------------------------------------------------------------
def clean_nepali_text(text) -> str:
    """data_prep.py:10-19 -- used by the URGENCY pipeline."""
    if pd.isna(text):
        return ""
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^A-Za-z0-9ऀ-ॿ\s.,!?;:()\"'-]", "", text)
    return text.strip()


def clean_text_dept(text) -> str:
    """preprocess_and_prepare_dataset.py:31-37 -- used by the DEPARTMENT pipeline."""
    text = re.sub(r"https?://\S+|www\.\S+", "", str(text))
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def norm(text) -> str:
    """Normalisation used for *provenance* and *leakage* matching only.

    strip().lower() with whitespace collapsed, per the audit's definition.
    This is exact-match after normalisation, so every leakage/overlap figure
    it produces is a LOWER BOUND (near-duplicates are not counted).
    """
    if pd.isna(text):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


# --------------------------------------------------------------------------
# Split reimplementations
# --------------------------------------------------------------------------
def urgency_splits(random_state: int = 42, test_size: float = 0.3, val_size: float = 0.5):
    """Reimplements data_prep.load_and_prepare_data. Returns (train, val, test)."""
    df = pd.read_csv(CORPUS_6000)
    df["clean_text"] = df["grievance"].apply(clean_nepali_text)
    df = df[df["clean_text"].str.len() > 10].reset_index(drop=True)
    df["label"] = df["urgency"].map(URGENCY2ID)
    train_df, temp_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state)
    val_df, test_df = train_test_split(
        temp_df, test_size=val_size, stratify=temp_df["label"], random_state=random_state)
    return train_df, val_df, test_df


def dept_splits(random_state: int = 42, train_size: float = 0.8,
                test_size: float = 0.1, val_size: float = 0.1):
    """Reimplements preprocess_and_prepare_dataset.split_dataset. Returns (train, eval, test)."""
    df = pd.read_csv(CORPUS_6000).copy()
    df["grievance"] = df["grievance"].apply(clean_text_dept)
    df["label"] = df["department"].map(DEPARTMENT2ID)
    df = df[["grievance", "label"]].dropna()

    temp_ratio = val_size + test_size
    train_df, temp_df = train_test_split(
        df, test_size=temp_ratio, random_state=random_state, stratify=df["label"])
    relative_val_size = val_size / temp_ratio
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - relative_val_size), random_state=random_state,
        stratify=temp_df["label"])
    return train_df, val_df, test_df


# --------------------------------------------------------------------------
# Provenance stratification of the 6,000-row corpus
# --------------------------------------------------------------------------
def provenance_strata() -> pd.DataFrame:
    """Assign every row of the 6,000-row corpus to a provenance stratum.

    A -- Hello Sarkar (real Nepal portal), government `complain_type` gold labels
    B -- Indian civic tweets, Llama-3 pseudo-labels
    C -- English 'categorized grievances', hand-labelled
    D -- untraceable to any raw file in the repository

    Matching is by normalised text against the 1,640-row intermediate corpus,
    whose rows carry `id`s that join to the raw files. The 6,000-row corpus has
    synthetic sequential ids (83223-89222) that join to nothing, so text
    matching is the only available route.
    """
    big = pd.read_csv(CORPUS_6000)
    mid = pd.read_csv(CORPUS_1640)
    details = pd.read_csv(RAW_DETAILS)
    categ = pd.read_csv(RAW_CATEG)

    hello_ids = set(details["id"].dropna().astype("int64"))
    categ_norm = {norm(t) for t in categ["Description"].dropna()}

    mid = mid.copy()
    mid["nt"] = mid["grievance"].map(norm)
    mid["stratum"] = "B"                                  # tweets are the residual
    mid.loc[mid["nt"].isin(categ_norm), "stratum"] = "C"
    mid.loc[mid["id"].isin(hello_ids), "stratum"] = "A"   # gold wins over any other tag

    a_set = set(mid.loc[mid.stratum == "A", "nt"])
    b_set = set(mid.loc[mid.stratum == "B", "nt"])
    c_set = set(mid.loc[mid.stratum == "C", "nt"])

    big = big.copy()
    big["nt"] = big["grievance"].map(norm)
    big["stratum"] = "D"
    big.loc[big["nt"].isin(b_set), "stratum"] = "B"
    big.loc[big["nt"].isin(c_set), "stratum"] = "C"
    big.loc[big["nt"].isin(a_set), "stratum"] = "A"
    return big


# --------------------------------------------------------------------------
# Leakage
# --------------------------------------------------------------------------
def leakage(train_texts, test_texts) -> dict:
    """Fraction of test rows whose normalised text also occurs in train."""
    tr = {norm(t) for t in train_texts}
    hits = [norm(t) in tr for t in test_texts]
    n = len(hits)
    k = int(np.sum(hits))
    return {"n_test": n, "n_leaked": k, "pct": round(100.0 * k / n, 4) if n else None,
            "mask": np.array(hits)}


# --------------------------------------------------------------------------
# The reference linear baseline
# --------------------------------------------------------------------------
def tfidf_char_logreg(seed: int = 42):
    """The baseline configuration used throughout the audit and this evidence pack."""
    return (
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=5, random_state=seed),
    )


def fit_eval(train_texts, train_y, evals: dict, seed: int = 42) -> dict:
    """Fit the linear baseline and score it on each named evaluation set.

    `evals` maps name -> (texts, y). Returns {name: {acc, macro_f1, n, y_pred}}.
    """
    vec, clf = tfidf_char_logreg(seed)
    X = vec.fit_transform(train_texts)
    clf.fit(X, train_y)
    out = {}
    for name, (texts, y) in evals.items():
        if len(texts) == 0:
            out[name] = {"n": 0, "acc": None, "macro_f1": None, "y_pred": []}
            continue
        pred = clf.predict(vec.transform(texts))
        out[name] = {
            "n": int(len(y)),
            "acc": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
            "y_pred": [int(p) for p in pred],
        }
    return out


# --------------------------------------------------------------------------
# Bootstrap CI on a fixed prediction vector
# --------------------------------------------------------------------------
def bootstrap_ci(y_true, y_pred, n_boot: int = 2000, seed: int = 12345, alpha: float = 0.05):
    """Percentile bootstrap CI for accuracy and macro-F1 over the test set.

    Resamples the (y_true, y_pred) pairs, so it quantifies evaluation-set
    sampling uncertainty for a FIXED trained model. It does not capture
    training-set variability -- that is what the seed sweep is for.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    accs = np.empty(n_boot)
    f1s = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_pred[idx]
        accs[b] = accuracy_score(yt, yp)
        f1s[b] = f1_score(yt, yp, average="macro", zero_division=0)
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "acc_mean": float(accs.mean()),
        "acc_ci95": [float(np.percentile(accs, lo)), float(np.percentile(accs, hi))],
        "macro_f1_mean": float(f1s.mean()),
        "macro_f1_ci95": [float(np.percentile(f1s, lo)), float(np.percentile(f1s, hi))],
        "n_boot": n_boot,
        "n_test": int(n),
    }
