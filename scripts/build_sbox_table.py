import glob
import json
import time
import warnings
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from llamea.ast_features import extract_ast_features

warnings.filterwarnings("ignore", category=RuntimeWarning, module="networkx")

# always resolve the project root
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "sbox_features_cache.pkl"
OUT = ROOT / "data" / "sbox_table.csv"

rows = []
for log_path in glob.glob(str(ROOT / "data/raw/SBOX/*/log.jsonl")):
    run_id = Path(log_path).parent.name  
    with open(log_path) as f:
        for line in f:
            record = json.loads(line)
            record["run"] = run_id
            rows.append(record)

df = pd.DataFrame(rows)

df["fitness"] = pd.to_numeric(df["fitness"], errors="coerce")

df["no_code"] = df["feedback"].fillna("").str.contains(
    "Could not extract generated code", regex=False
)

valid = df[np.isfinite(df["fitness"])].copy()
n_total_valid = len(valid)  # remember the full count before sampling
print("Valid rows:", len(valid), "of", len(df))

# extract features (or load from cache)
def safe_features(code):
    try:
        return extract_ast_features(code)
    except Exception as e:
        print("Error extracting features:", e)
        return None

if CACHE.exists():
    print("Loading features from cache:", CACHE)
    valid["features"] = pd.read_pickle(CACHE)
else:
    tqdm.pandas()
    start = time.time()
    valid["features"] = valid["code"].progress_apply(safe_features)
    print(f"Extraction took {(time.time() - start) / 60:.1f} min")
    valid["features"].to_pickle(CACHE)
    print("Cached features to", CACHE)

n_fail = valid["features"].isna().sum()
print("Feature extraction failed:", n_fail, "of", len(valid))

valid = valid[valid["features"].notna()]
feats = pd.DataFrame(valid["features"].tolist(), index=valid.index)

print("Number of features:", feats.shape[1])

# combine id columns and features side-by-side
id_cols = ["id", "run", "generation", "fitness", "parent_ids", "code"]
missing = [c for c in id_cols if c not in valid.columns]
if missing:
    print("Warning: these columns don't exist and were skipped:", missing)
id_cols = [c for c in id_cols if c in valid.columns]

table = pd.concat([valid[id_cols], feats], axis=1)
print("Table shape:", table.shape)

# check for nan and inf
nan_counts = feats.isna().sum()
inf_counts = np.isinf(feats).sum()
print("NaN per column:\n", nan_counts[nan_counts > 0])
print("inf per column:\n", inf_counts[inf_counts > 0])
print("Constant features:", feats.columns[feats.nunique() <= 1].tolist())

# save nan/inf left as-is to be handled at training time
table.to_csv(OUT, index=False)
print("Saved", OUT)
