import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
import warnings
warnings.filterwarnings('ignore')

# ========================================================
# 1. データの読み込みと極限の特徴量エンジニアリング
# ========================================================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

def create_ultimate_features(df):
    df_out = df.copy()
    
    # --- 基本特徴量 ---
    df_out['age'] = 2026 - df_out['birth_year']
    df_out['total_children'] = df_out['num_children'] + df_out['num_teenagers']
    df_out['has_partner'] = df_out['marital_status'].isin(['Married', 'Partner']).astype(int)
    df_out['family_size'] = 1 + df_out['has_partner'] + df_out['total_children']
    
    # --- 消費・購買の集計 ---
    spend_cols = ['spend_wines', 'spend_fruits', 'spend_meat', 'spend_fish', 'spend_sweets', 'spend_gold']
    purchase_cols = ['deals_purchases', 'web_purchases', 'catalog_purchases', 'store_purchases']
    
    df_out['total_spend'] = df_out[spend_cols].sum(axis=1)
    df_out['total_purchases'] = df_out[purchase_cols].sum(axis=1)
    
    # --- 新規：ゼロ消費フラグ（全く買わないカテゴリがあるか） ---
    for col in spend_cols:
        df_out[f'is_zero_{col}'] = (df_out[col] == 0).astype(int)
        
    # --- 比率・割り算系の特徴量 ---
    df_out['spend_income_ratio'] = df_out['total_spend'] / (df_out['annual_income'] + 1e-5)
    df_out['spend_per_family_member'] = df_out['total_spend'] / df_out['family_size']
    df_out['income_per_age'] = df_out['annual_income'] / df_out['age']
    
    # --- 日付処理 ---
    if 'registration_date' in df_out.columns:
        df_out['registration_date'] = pd.to_datetime(df_out['registration_date'])
        base_date = pd.to_datetime('2026-01-01')
        df_out['days_since_registration'] = (base_date - df_out['registration_date']).dt.days
        df_out.drop(['registration_date'], axis=1, inplace=True)
        
    return df_out

train_feat = create_ultimate_features(train_df)
test_feat = create_ultimate_features(test_df)

# ========================================================
# 2. アンサンブル学習（LightGBM + XGBoost + CatBoost）
# ========================================================
drop_cols = ['customer_id', 'target']
cat_cols = ['education_level', 'marital_status']

X = train_feat.drop(columns=[col for col in drop_cols if col in train_feat.columns])
y = train_feat['target']
X_test = test_feat.drop(columns=[col for col in ['customer_id'] if col in test_feat.columns])

# カテゴリ変数の型変換（アルゴリズムによって要求される型が違うため調整）
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# 各モデルのOOFとテスト予測を格納する配列
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_preds_lgb = np.zeros(len(X_test))
test_preds_xgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

print("--- マルチモデル・アンサンブル学習を開始します ---")

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
    print(f"\n========== Fold {fold+1} ==========")
    
    # --- データ分割 ---
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[valid_idx], y.iloc[valid_idx]
    
    X_tr_lgb, X_va_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[valid_idx]
    
    # --- Target Encoding (XGBoost等のため、Fold内で行うことでデータ漏洩を防ぐ) ---
    te = TargetEncoder(cols=cat_cols)
    X_tr_enc = te.fit_transform(X_tr, y_tr)
    X_va_enc = te.transform(X_va)
    X_test_enc = te.transform(X_test)
    
    # ----------------------------------------------------
    # Model 1: LightGBM
    # ----------------------------------------------------
    dtrain_lgb = lgb.Dataset(X_tr_lgb, label=y_tr)
    dvalid_lgb = lgb.Dataset(X_va_lgb, label=y_va, reference=dtrain_lgb)
    
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.03,
        'max_depth': 6, 'num_leaves': 31, 'colsample_bytree': 0.8,
        'subsample': 0.8, 'random_state': 42, 'verbosity': -1
    }
    model_lgb = lgb.train(
        lgb_params, dtrain_lgb, valid_sets=[dvalid_lgb],
        num_boost_round=3000, callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    oof_lgb[valid_idx] = model_lgb.predict(X_va_lgb)
    test_preds_lgb += model_lgb.predict(X_test_lgb) / n_splits
    print(f"[LightGBM] Fold {fold+1} AUC: {roc_auc_score(y_va, oof_lgb[valid_idx]):.5f}")

    # ----------------------------------------------------
    # Model 2: XGBoost
    # ----------------------------------------------------
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'learning_rate': 0.03,
        'max_depth': 5, 'colsample_bytree': 0.8, 'subsample': 0.8, 'random_state': 42
    }
    dtrain_xgb = xgb.DMatrix(X_tr_enc, label=y_tr)
    dvalid_xgb = xgb.DMatrix(X_va_enc, label=y_va)
    dtest_xgb = xgb.DMatrix(X_test_enc)
    
    model_xgb = xgb.train(
        xgb_params, dtrain_xgb, evals=[(dvalid_xgb, 'eval')],
        num_boost_round=3000, early_stopping_rounds=100, verbose_eval=False
    )
    oof_xgb[valid_idx] = model_xgb.predict(dvalid_xgb)
    test_preds_xgb += model_xgb.predict(dtest_xgb) / n_splits
    print(f"[XGBoost]  Fold {fold+1} AUC: {roc_auc_score(y_va, oof_xgb[valid_idx]):.5f}")

    # ----------------------------------------------------
    # Model 3: CatBoost
    # ----------------------------------------------------
    model_cat = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=6,
        eval_metric='AUC', random_seed=42, verbose=False
    )
    model_cat.fit(
        X_tr, y_tr, cat_features=cat_cols,
        eval_set=(X_va, y_va), early_stopping_rounds=100
    )
    oof_cat[valid_idx] = model_cat.predict_proba(X_va)[:, 1]
    test_preds_cat += model_cat.predict_proba(X_test)[:, 1] / n_splits
    print(f"[CatBoost] Fold {fold+1} AUC: {roc_auc_score(y_va, oof_cat[valid_idx]):.5f}")

# ========================================================
# 3. アンサンブルの実行とスコア確認
# ========================================================
print("\n" + "="*50)
print(f"★ LightGBM Overall AUC : {roc_auc_score(y, oof_lgb):.5f}")
print(f"★ XGBoost  Overall AUC : {roc_auc_score(y, oof_xgb):.5f}")
print(f"★ CatBoost Overall AUC : {roc_auc_score(y, oof_cat):.5f}")

# 3つのモデルの予測をブレンド（単純平均）
oof_ensemble = (oof_lgb + oof_xgb + oof_cat) / 3.0
ensemble_auc = roc_auc_score(y, oof_ensemble)

print("-" * 50)
print(f"★★★ Ensemble CV Overall AUC: {ensemble_auc:.5f} ★★★")
print("-" * 50)

# テスト予測もブレンドして提出ファイル作成
test_preds_ensemble = (test_preds_lgb + test_preds_xgb + test_preds_cat) / 3.0

submission = pd.DataFrame({
    'customer_id': test_df['customer_id'],
    'target': test_preds_ensemble
})
submission.to_csv("submission_ensemble_ultimate.csv", index=False)
print("予測が完了しました！ `submission_ensemble_ultimate.csv` を提出してみてください。")