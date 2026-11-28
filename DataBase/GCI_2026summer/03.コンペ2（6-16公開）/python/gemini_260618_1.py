import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

# ========================================================
# 1. データの読み込みと「最も安定した」特徴量エンジニアリング
# ========================================================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

def create_best_features(df):
    df_out = df.copy()
    
    # 属性特徴量
    df_out['age'] = 2026 - df_out['birth_year']
    df_out['total_children'] = df_out['num_children'] + df_out['num_teenagers']
    df_out['has_partner'] = df_out['marital_status'].isin(['Married', 'Partner']).astype(int)
    df_out['family_size'] = 1 + df_out['has_partner'] + df_out['total_children']
    
    # 消費・購買の集計
    spend_cols = ['spend_wines', 'spend_fruits', 'spend_meat', 'spend_fish', 'spend_sweets', 'spend_gold']
    purchase_cols = ['deals_purchases', 'web_purchases', 'catalog_purchases', 'store_purchases']
    
    df_out['total_spend'] = df_out[spend_cols].sum(axis=1)
    df_out['total_purchases'] = df_out[purchase_cols].sum(axis=1)
    
    # ゼロ消費フラグ
    for col in spend_cols:
        df_out[f'is_zero_{col}'] = (df_out[col] == 0).astype(int)
        
    # 高度な比率
    df_out['spend_income_ratio'] = df_out['total_spend'] / (df_out['annual_income'] + 1e-5)
    df_out['spend_per_family_member'] = df_out['total_spend'] / df_out['family_size']
    df_out['income_per_age'] = df_out['annual_income'] / df_out['age']
    
    # RFMの簡易版
    df_out['avg_spend_per_purchase'] = df_out['total_spend'] / (df_out['total_purchases'] + 1e-5)
    
    if 'registration_date' in df_out.columns:
        df_out['registration_date'] = pd.to_datetime(df_out['registration_date'])
        base_date = pd.to_datetime('2026-01-01')
        df_out['days_since_registration'] = (base_date - df_out['registration_date']).dt.days
        df_out.drop(['registration_date'], axis=1, inplace=True)
        
    return df_out

train_feat = create_best_features(train_df)
test_feat = create_best_features(test_df)

drop_cols = ['customer_id', 'target']
cat_cols = ['education_level', 'marital_status']

X = train_feat.drop(columns=[col for col in drop_cols if col in train_feat.columns])
y = train_feat['target']
X_test = test_feat.drop(columns=[col for col in ['customer_id'] if col in test_feat.columns])

X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

# ========================================================
# 2. ベースモデルの学習（OOF予測の取得）
# ========================================================
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb, oof_xgb, oof_cat = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
test_lgb, test_xgb, test_cat = np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))

print("--- モデル学習開始 ---")
for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[valid_idx], y.iloc[valid_idx]
    X_tr_lgb, X_va_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[valid_idx]
    
    te = TargetEncoder(cols=cat_cols)
    X_tr_enc = te.fit_transform(X_tr, y_tr)
    X_va_enc = te.transform(X_va)
    X_test_enc = te.transform(X_test)
    
    # LightGBM
    dtrain_lgb = lgb.Dataset(X_tr_lgb, label=y_tr)
    dvalid_lgb = lgb.Dataset(X_va_lgb, label=y_va, reference=dtrain_lgb)
    model_lgb = lgb.train({'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.03, 'max_depth': 6, 'verbosity': -1, 'random_state': 42},
                          dtrain_lgb, valid_sets=[dvalid_lgb], num_boost_round=3000, callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[valid_idx] = model_lgb.predict(X_va_lgb)
    test_lgb += model_lgb.predict(X_test_lgb) / n_splits
    
    # XGBoost
    dtrain_xgb = xgb.DMatrix(X_tr_enc, label=y_tr)
    dvalid_xgb = xgb.DMatrix(X_va_enc, label=y_va)
    model_xgb = xgb.train({'objective': 'binary:logistic', 'eval_metric': 'auc', 'learning_rate': 0.03, 'max_depth': 5, 'random_state': 42},
                          dtrain_xgb, evals=[(dvalid_xgb, 'eval')], num_boost_round=3000, early_stopping_rounds=100, verbose_eval=False)
    oof_xgb[valid_idx] = model_xgb.predict(dvalid_xgb)
    test_xgb += model_xgb.predict(xgb.DMatrix(X_test_enc)) / n_splits
    
    # CatBoost
    model_cat = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, eval_metric='AUC', random_seed=42, verbose=False)
    model_cat.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va), early_stopping_rounds=100)
    oof_cat[valid_idx] = model_cat.predict_proba(X_va)[:, 1]
    test_cat += model_cat.predict_proba(X_test)[:, 1] / n_splits

print("\n--- 各モデルの単体CV AUC ---")
print(f"LightGBM : {roc_auc_score(y, oof_lgb):.5f}")
print(f"XGBoost  : {roc_auc_score(y, oof_xgb):.5f}")
print(f"CatBoost : {roc_auc_score(y, oof_cat):.5f}")

# ========================================================
# 3. 最適なブレンド比率の探索（最適化アルゴリズム）
# ========================================================
print("\n--- 最適なブレンド比率を探索中... ---")
# 目的関数（AUCを最大化するために、マイナスAUCを最小化する）
def objective_func(weights):
    weights = np.clip(weights, 0, 1) # 重みは0~1の範囲
    weights /= np.sum(weights) # 合計を1にする
    blended_oof = (weights[0] * oof_lgb) + (weights[1] * oof_xgb) + (weights[2] * oof_cat)
    return -roc_auc_score(y, blended_oof)

# 初期重み（均等）からスタート
init_weights = [1/3, 1/3, 1/3]
bounds = [(0, 1), (0, 1), (0, 1)]

# 最適化の実行
res = minimize(objective_func, init_weights, method='Nelder-Mead', bounds=bounds)

# 最適な重みを正規化
best_weights = np.clip(res.x, 0, 1)
best_weights /= np.sum(best_weights)

print(f"\n★ 発見された最強の配合比率 ★")
print(f"LightGBM: {best_weights[0]:.3f}, XGBoost: {best_weights[1]:.3f}, CatBoost: {best_weights[2]:.3f}")

# ========================================================
# 4. 最終予測の作成
# ========================================================
best_oof = (best_weights[0] * oof_lgb) + (best_weights[1] * oof_xgb) + (best_weights[2] * oof_cat)
best_auc = roc_auc_score(y, best_oof)

print("=" * 50)
print(f"★★★ 最終 Optimized Blending CV AUC: {best_auc:.5f} ★★★")
print("=" * 50)

# テストデータにも同じ比率を適用
final_test_preds = (best_weights[0] * test_lgb) + (best_weights[1] * test_xgb) + (best_weights[2] * test_cat)

submission = pd.DataFrame({
    'customer_id': test_df['customer_id'],
    'target': final_test_preds
})
submission.to_csv("submission_optimized_blend.csv", index=False)
print("予測が完了しました！ `submission_optimized_blend.csv` を提出してみてください。")