import glob
import json
from pathlib import Path
import pandas as pd
import numpy as np
from llamea.ast_features import extract_ast_features

# last updated: 3 oct
# next steps: extract_ast_features from full dataset and build the table
rows = []
for log_path in glob.glob("data/raw/SBOX/*/log.jsonl"):
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
print("Valid rows:", len(valid), "of", len(df))

valid = valid.sample(10, random_state=0) # just to test, only use 10 programs

def safe_features(code):
    try:
        return extract_ast_features(code)
    except Exception as e:
        print("Error extracting features:", e)
        return None

valid["features"] = valid["code"].apply(safe_features)

n_fail = valid["features"].isna().sum()
print("Feature extraction failed:", n_fail, "of", len(valid))

feats = pd.DataFrame(valid["features"].tolist(), index=valid.index)

print("Features shape:", feats.shape[1])
print(feats.columns.tolist())

