from pathlib import Path

from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.dummy import DummyRegressor
from xgboost import XGBRegressor

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLE = ROOT / "data" / "sbox_table.csv"

NON_FEATURE_COLS = ["id", "run", "generation", "fitness", "parent_ids", "code"]

table = pd.read_csv(TABLE)
print("Table shape:", table.shape)

# features are everything that isn't an identifier or the target
X = table.drop(columns=[c for c in NON_FEATURE_COLS if c in table.columns])
y = table["fitness"]
groups = table["run"]

# drop the useless features
DROP_CONSTANT = True
if DROP_CONSTANT:
    constant = X.columns[X.nunique() <= 1].tolist()
    X = X.drop(columns=constant)
    print("Dropped constant features:", constant)


dummy = DummyRegressor(strategy="mean")
preds = cross_val_predict(dummy, X, y, cv=GroupKFold(5), groups=groups)
print("Dummy R2 (GroupKFold):", r2_score(y, preds))

# model = XGBRegressor(
#     max_depth=4,            # base was 4, found it increased as it went down
#     n_estimators=200,
#     learning_rate=0.1,
#     random_state=0,
#     n_jobs=-1,
# )

# splits = {
#     "random (KFold)": dict(cv=KFold(5, shuffle=True, random_state=0)),
#     "grouped by run (GroupKFold)": dict(cv=GroupKFold(5), groups=groups),
# }

# results = {}
# for name, kw_args in splits.items():
#     preds = cross_val_predict(model, X, y, **kw_args)
#     results[name] = {
#         "R2": r2_score(y, preds),
#         "Spearman": spearmanr(y, preds)[0],
#     }

# print(pd.DataFrame(results).T.round(3))

# run_means = table.groupby("run")["fitness"].mean()
# print("Between-run variance:", run_means.var())
# print("Overall variance:", table["fitness"].var())
# print("Ratio:", run_means.var() / table["fitness"].var())


# X with run test
# X_with_run = X.copy()
# X_with_run["run_id"] = table["run"].astype("category").cat.codes
# preds = cross_val_predict(model, X_with_run, y, cv=KFold(5, shuffle=True, random_state=0))
# print("Random KFold R2 with run_id:", r2_score(y, preds))

"""
run more experiments with different random seeds for the cross validation of the splitting
report the standard deviation 
add storage of every result for repetition
run the XGBoost with teh default paraemters except random_State keep it 0
autoGluon, try this in addition to XGBoost 
find an automation framework with scikit-learn to make this easier
consider training using parents for later children??

in addition: may need a custom ranking metric as we only care about the top-k performers instead of spearman
for more information to be reported

ranking > accuracy

new metric extraction function for the more trivial features of algorithms like number of lines of code
cyclo complexity, functions, loops.etc ? for speed

can use multiple datasets with the SBOX dataset
eventually add graphs + visual presentation with the different approaches used

order of execution:
embeddings -> pipeline improvement -> ??
"""