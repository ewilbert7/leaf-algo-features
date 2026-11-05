import glob
import hashlib
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
FAILURES_OUT = ROOT / "data" / "sbox_failures.csv"

# e.g. run-LLaMEA-gemini-2.0-flash-SBOX_COST_fid21-4
RUN_NAME = r"^run-LLaMEA-(?P<llm>.+)-SBOX_COST_(?P<benchmark>.+)-(?P<repeat>\d+)$"

rows = []
# sorted so the row order doesn't depend on the filesystem
for log_path in sorted(glob.glob(str(ROOT / "data/raw/SBOX/*/log.jsonl"))):
    run_id = Path(log_path).parent.name  
    with open(log_path) as f:
        for line in f:
            record = json.loads(line)
            record["run"] = run_id
            rows.append(record)

df = pd.DataFrame(rows)

run_parts = df["run"].str.extract(RUN_NAME)
if run_parts.isna().any(axis=None):
    raise ValueError(f"Unexpected run names: {df.loc[run_parts.isna().any(axis=1), 'run'].unique()}")
df = pd.concat([df, run_parts], axis=1)

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

def code_key(code):
    return hashlib.sha256(code.encode()).hexdigest()

# the cache is keyed by a hash of the code, so a feature can't end up on the
# wrong program if the rows change order or the raw data changes
cache = {}
if CACHE.exists():
    loaded = pd.read_pickle(CACHE)
    if isinstance(loaded, dict):
        print("Loading features from cache:", CACHE)
        cache = loaded
    else:
        print("Ignoring old-format cache, re-extracting:", CACHE)

keys = valid["code"].map(code_key)
to_extract = valid.loc[~keys.isin(cache)].drop_duplicates("code")["code"]
if len(to_extract) > 0:
    start = time.time()
    for code in tqdm(to_extract):
        cache[code_key(code)] = safe_features(code)
    print(f"Extraction took {(time.time() - start) / 60:.1f} min")
    pd.to_pickle(cache, CACHE)
    print("Cached features to", CACHE)

valid["features"] = [cache[key] for key in keys]

n_fail = valid["features"].isna().sum()
print("Feature extraction failed:", n_fail, "of", len(valid))

# failed programs stay out of the fitness table but are kept here so that
# failure can be modelled separately later
failed_eval = df[~np.isfinite(df["fitness"])].copy()
failed_eval["failure_type"] = np.where(failed_eval["no_code"], "no_code", "evaluation_error")
failed_features = valid[valid["features"].isna()].copy()
failed_features["failure_type"] = "feature_extraction"
failed = pd.concat([failed_eval, failed_features])
failure_cols = ["id", "run", "llm", "benchmark", "repeat", "generation", "failure_type", "feedback", "code"]
failed[failure_cols].to_csv(FAILURES_OUT, index=False)
print("Failure types:\n", failed["failure_type"].value_counts())
print("Saved", FAILURES_OUT)

valid = valid[valid["features"].notna()]
feats = pd.DataFrame(valid["features"].tolist(), index=valid.index)

print("Number of features:", feats.shape[1])

# combine id columns and features side-by-side
id_cols = ["id", "run", "llm", "benchmark", "repeat", "generation", "fitness", "parent_ids", "code"]
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
