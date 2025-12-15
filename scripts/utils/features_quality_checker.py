import os
import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import entropy

# ==========================================
# CONFIG
# ==========================================
FEATURE_ROOT = "./data/features_v3"   # change if needed
EXPECTED_DIM = 960
MIN_FRAMES = 5

OUTPUT_CSV = "feature_quality_report.csv"

# ==========================================
# UTILS
# ==========================================
def compute_entropy(features):
    """
    Compute mean entropy over feature dimensions
    """
    eps = 1e-8
    probs = np.abs(features)
    probs = probs / (np.sum(probs, axis=1, keepdims=True) + eps)
    ent = entropy(probs, axis=1)
    return float(np.mean(ent))

# ==========================================
# MAIN CHECKER
# ==========================================
rows = []

feature_files = glob.glob(os.path.join(FEATURE_ROOT, "**", "*.npy"), recursive=True)

print(f"[INFO] Found {len(feature_files)} feature files")

for fpath in tqdm(feature_files, desc="Checking features"):
    row = {
        "feature_path": fpath,
        "video_name": os.path.basename(fpath),
        "num_frames": 0,
        "feature_dim": None,
        "mean": None,
        "std": None,
        "min": None,
        "max": None,
        "entropy": None,
        "has_nan": False,
        "has_inf": False,
        "low_variance": False,
        "status": "OK"
    }

    try:
        feats = np.load(fpath)

        # Shape checks
        if feats.ndim != 2:
            row["status"] = "BAD_SHAPE"
            rows.append(row)
            continue

        num_frames, feat_dim = feats.shape
        row["num_frames"] = num_frames
        row["feature_dim"] = feat_dim

        if feat_dim != EXPECTED_DIM:
            row["status"] = "WRONG_DIM"
            rows.append(row)
            continue

        if num_frames < MIN_FRAMES:
            row["status"] = "TOO_SHORT"
            rows.append(row)
            continue

        # Numerical checks
        row["has_nan"] = bool(np.isnan(feats).any())
        row["has_inf"] = bool(np.isinf(feats).any())

        if row["has_nan"] or row["has_inf"]:
            row["status"] = "NAN_OR_INF"
            rows.append(row)
            continue

        # Statistics
        row["mean"] = float(np.mean(feats))
        row["std"] = float(np.std(feats))
        row["min"] = float(np.min(feats))
        row["max"] = float(np.max(feats))

        if row["std"] < 1e-5:
            row["low_variance"] = True
            row["status"] = "LOW_VARIANCE"

        # Entropy
        row["entropy"] = compute_entropy(feats)

    except Exception as e:
        row["status"] = f"LOAD_ERROR: {str(e)}"

    rows.append(row)

# ==========================================
# SAVE CSV
# ==========================================
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_CSV, index=False)

print("\n[SUCCESS] Feature quality report saved as:")
print(f"👉 {OUTPUT_CSV}")

# ==========================================
# SUMMARY
# ==========================================
print("\n[SUMMARY]")
print(df["status"].value_counts())
