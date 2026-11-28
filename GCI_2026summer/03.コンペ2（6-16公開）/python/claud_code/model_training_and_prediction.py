"""
Customer campaign-response model
=================================
Train: train.csv (target = 1 customer responded to campaign / 0 did not)
Test : test.csv  (no target -> needs prediction)

Output: predictions.csv with columns [customer_id, target]
        target = predicted probability of response (needed for AUC evaluation)

Honesty note on AUC
--------------------
The brief asked for a model whose AUC, *after fully removing overfitting*,
exceeds 0.910. To check this honestly we never look at training-set AUC
(which is trivially inflate-able). Instead we use repeated stratified
k-fold cross-validation: each sample's prediction is always made by a model
that never saw that sample during training. That is the only way to get an
unbiased estimate of how the model will perform on the unseen test.csv.

With this dataset (1568 rows, 234 positives, and no campaign-history columns)
the honest cross-validated AUC plateaus at ~0.87-0.88 (std ~0.02) no matter
which model family / feature set / tuning we throw at it -- see the printed
CV report below. A handful of individual folds do exceed 0.91 (max observed
0.917), but the *average*, which is the trustworthy estimate, does not.
We report this transparently rather than presenting a cherry-picked fold or
an over-fit training-score as if it were 0.91+.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Feature engineering
# ----------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["registration_date"] = pd.to_datetime(df["registration_date"])
    ref_date = pd.Timestamp("2015-01-01")  # fixed reference, same for train & test

    # Obvious data-entry errors -> NaN (HistGradientBoosting handles NaN natively,
    # so we don't have to invent an imputed value and risk leaking train stats)
    df.loc[df["birth_year"] < 1920, "birth_year"] = np.nan
    df.loc[df["annual_income"] > 600_000, "annual_income"] = np.nan

    df["age"] = ref_date.year - df["birth_year"]
    df["tenure_days"] = (ref_date - df["registration_date"]).dt.days

    spend_cols = ["spend_wines", "spend_fruits", "spend_meat",
                  "spend_fish", "spend_sweets", "spend_gold"]
    df["total_spend"] = df[spend_cols].sum(axis=1)
    df["spend_gold_ratio"] = df["spend_gold"] / df["total_spend"].replace(0, np.nan)
    df["spend_meat_ratio"] = df["spend_meat"] / df["total_spend"].replace(0, np.nan)

    purchase_cols = ["deals_purchases", "web_purchases", "catalog_purchases", "store_purchases"]
    df["total_purchases"] = df[purchase_cols].sum(axis=1)

    df["num_dependents"] = df["num_children"] + df["num_teenagers"]
    couple = df["marital_status"].isin(["Married", "Partner"]).astype(int)
    df["family_size"] = 1 + couple + df["num_dependents"]
    df["income_per_capita"] = df["annual_income"] / df["family_size"]

    edu_map = {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}
    df["education_ordinal"] = df["education_level"].map(edu_map)

    return df


# Final feature list (selected via permutation importance + ablation -
# kept lean on purpose, extra noisy ratio/interaction columns measurably
# hurt the honest CV score on this small dataset)
FEATURES = [
    "days_since_last_purchase", "tenure_days", "monthly_web_visits",
    "catalog_purchases", "spend_wines", "store_purchases",
    "education_ordinal", "income_per_capita", "spend_gold_ratio",
    "spend_meat_ratio", "family_size", "total_spend", "spend_gold",
    "spend_meat", "spend_fish", "annual_income", "deals_purchases",
    "total_purchases",
]

BEST_PARAMS = dict(
    min_samples_leaf=30,
    max_leaf_nodes=10,
    max_iter=200,
    max_depth=None,
    learning_rate=0.1,
    l2_regularization=2,
    early_stopping=True,
    n_iter_no_change=20,
    validation_fraction=0.15,
    class_weight="balanced",
    random_state=RANDOM_STATE,
)


def build_model():
    return HistGradientBoostingClassifier(**BEST_PARAMS)


# ----------------------------------------------------------------------
# 2. Load & engineer
# ----------------------------------------------------------------------
train_raw = pd.read_csv("/mnt/user-data/uploads/train.csv")
test_raw = pd.read_csv("/mnt/user-data/uploads/test.csv")

train_fe = engineer_features(train_raw)
test_fe = engineer_features(test_raw)

X = train_fe[FEATURES]
y = train_fe["target"]
X_test = test_fe[FEATURES]

# ----------------------------------------------------------------------
# 3. Honest evaluation (out-of-fold predictions only)
# ----------------------------------------------------------------------
print("=" * 70)
print("HONEST (out-of-fold) CROSS-VALIDATION REPORT")
print("=" * 70)

skf5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
fold_scores = cross_val_score(build_model(), X, y, cv=skf5, scoring="roc_auc")
for i, s in enumerate(fold_scores, 1):
    print(f"  Fold {i}: AUC = {s:.4f}")
print(f"  5-fold mean AUC = {fold_scores.mean():.4f}  (std {fold_scores.std():.4f})")

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RANDOM_STATE)
rep_scores = cross_val_score(build_model(), X, y, cv=rskf, scoring="roc_auc", n_jobs=-1)
print(f"\n  Repeated CV (5-fold x 10 repeats, 50 folds total):")
print(f"    mean AUC = {rep_scores.mean():.4f}")
print(f"    std      = {rep_scores.std():.4f}")
print(f"    min/max  = {rep_scores.min():.4f} / {rep_scores.max():.4f}")

print("\n  -> This is the trustworthy, leakage-free estimate of generalization")
print("     performance. Individual lucky folds can clear 0.910, but the")
print("     honest *average* across many folds is what should be trusted.")

# ----------------------------------------------------------------------
# 4. Fit final model on all training data & predict test.csv
# ----------------------------------------------------------------------
final_model = build_model()
final_model.fit(X, y)

test_proba = final_model.predict_proba(X_test)[:, 1]

submission = pd.DataFrame({
    "customer_id": test_fe["customer_id"],
    "target": test_proba,
})
out_path = "/mnt/user-data/outputs/predictions.csv"
submission.to_csv(out_path, index=False)
print(f"\nSaved predictions for {len(submission)} customers to {out_path}")
print(submission.head())
