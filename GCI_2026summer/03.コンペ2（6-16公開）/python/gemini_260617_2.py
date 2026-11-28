import pandas as pd
import numpy as np
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ========================================================
# 1. データの読み込みと高度な特徴量エンジニアリング
# ========================================================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

def create_advanced_features(df):
    df_out = df.copy()
    
    # 1. 基本的な特徴量
    df_out['age'] = 2026 - df_out['birth_year']
    df_out['total_children'] = df_out['num_children'] + df_out['num_teenagers']
    
    # 配偶者フラグ（Partner, Marriedなら1とする簡易的な処理）
    df_out['has_partner'] = df_out['marital_status'].isin(['Married', 'Partner']).astype(int)
    # 推定家族構成人数（本人(1) + 配偶者 + 子供）
    df_out['family_size'] = 1 + df_out['has_partner'] + df_out['total_children']

    # 2. 消費・購買系の集計
    spend_cols = ['spend_wines', 'spend_fruits', 'spend_meat', 'spend_fish', 'spend_sweets', 'spend_gold']
    purchase_cols = ['deals_purchases', 'web_purchases', 'catalog_purchases', 'store_purchases']
    
    df_out['total_spend'] = df_out[spend_cols].sum(axis=1)
    df_out['total_purchases'] = df_out[purchase_cols].sum(axis=1)
    
    # 3. 高度な比率特徴量（0除算回避のため 1e-5 を加算）
    # 年収に対する消費割合（お金に余裕があるか・スーパーにどれだけお金を落としているか）
    df_out['spend_income_ratio'] = df_out['total_spend'] / (df_out['annual_income'] + 1e-5)
    
    # 家族1人あたりの消費額
    df_out['spend_per_family_member'] = df_out['total_spend'] / df_out['family_size']
    
    # 特定チャネルの利用割合（Web偏重か、店舗偏重か）
    df_out['web_purchase_ratio'] = df_out['web_purchases'] / (df_out['total_purchases'] + 1e-5)
    df_out['store_purchase_ratio'] = df_out['store_purchases'] / (df_out['total_purchases'] + 1e-5)
    
    # 日付処理（登録からの経過日数）
    if 'registration_date' in df_out.columns:
        df_out['registration_date'] = pd.to_datetime(df_out['registration_date'])
        base_date = pd.to_datetime('2026-01-01')
        df_out['days_since_registration'] = (base_date - df_out['registration_date']).dt.days
        df_out.drop(['registration_date'], axis=1, inplace=True)
        
    return df_out

train_feat = create_advanced_features(train_df)
test_feat = create_advanced_features(test_df)

# カテゴリ変数の処理
cat_cols = ['education_level', 'marital_status']
for col in cat_cols:
    train_feat[col] = train_feat[col].astype('category')
    test_feat[col] = test_feat[col].astype('category')

drop_cols = ['customer_id', 'target']
X = train_feat.drop(columns=[col for col in drop_cols if col in train_feat.columns])
y = train_feat['target']
X_test = test_feat.drop(columns=[col for col in ['customer_id'] if col in test_feat.columns])

# ========================================================
# 2. Optunaによるハイパーパラメータの自動探索
# ========================================================
def objective(trial):
    """Optunaで最適化するための目的関数"""
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []

    for train_idx, valid_idx in cv.split(X, y):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
        
        dtrain = lgb.Dataset(X_train, label=y_train)
        dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
        
        model = lgb.train(
            params,
            dtrain,
            valid_sets=[dvalid],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
        )
        
        preds = model.predict(X_valid, num_iteration=model.best_iteration)
        auc_scores.append(roc_auc_score(y_valid, preds))
        
    return np.mean(auc_scores)

print("--- Optunaによるパラメータ探索を開始します（数十回実行されます） ---")
study = optuna.create_study(direction='maximize')
# 時間に余裕があれば n_trials=50 や 100 に増やすとより良いパラメータが見つかります
study.optimize(objective, n_trials=30) 

print(f"★ 最適なパラメータが見つかりました (Best AUC: {study.best_value:.5f})")
best_params = study.best_params
best_params.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'verbosity': -1})

# ========================================================
# 3. 最適パラメータでの最終学習と予測
# ========================================================
print("--- 最適パラメータを使用して全Foldで学習を行います ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    
    model = lgb.train(
        params=best_params,
        train_set=dtrain,
        num_boost_round=5000,
        valid_sets=[dtrain, dvalid],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]
    )
    
    valid_preds = model.predict(X_valid, num_iteration=model.best_iteration)
    oof_preds[valid_idx] = valid_preds
    fold_auc = roc_auc_score(y_valid, valid_preds)
    print(f"Fold {fold+1} AUC: {fold_auc:.5f}")
    
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / 5

final_cv_auc = roc_auc_score(y, oof_preds)
print("-" * 50)
print(f"★★ 最終 CV Overall AUC: {final_cv_auc:.5f} ★★")
print("-" * 50)

submission = pd.DataFrame({
    'customer_id': test_df['customer_id'],
    'target': test_preds
})
submission.to_csv("submission_lgbm_optuna.csv", index=False)