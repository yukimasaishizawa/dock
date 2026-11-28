import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# ========================================================
# 1. データの読み込みと高度な特徴量生成
# ========================================================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

def create_master_features(df, is_train=True):
    df_out = df.copy()
    
    # --- 1. 基本属性 ---
    df_out['age'] = 2026 - df_out['birth_year']
    df_out['total_children'] = df_out['num_children'] + df_out['num_teenagers']
    df_out['has_partner'] = df_out['marital_status'].isin(['Married', 'Partner']).astype(int)
    df_out['family_size'] = 1 + df_out['has_partner'] + df_out['total_children']
    
    # --- 2. RFM (Recency, Frequency, Monetary) ---
    spend_cols = ['spend_wines', 'spend_fruits', 'spend_meat', 'spend_fish', 'spend_sweets', 'spend_gold']
    purchase_cols = ['deals_purchases', 'web_purchases', 'catalog_purchases', 'store_purchases']
    
    df_out['Monetary'] = df_out[spend_cols].sum(axis=1) # 購買金額
    df_out['Frequency'] = df_out[purchase_cols].sum(axis=1) # 購買頻度
    df_out['Recency'] = df_out['days_since_last_purchase'] # 最新購入からの日数
    
    # RFMベースの派生特徴量（1回あたりの購入額、日数あたりの購入頻度）
    df_out['monetary_per_frequency'] = df_out['Monetary'] / (df_out['Frequency'] + 1e-5)
    df_out['frequency_per_recency'] = df_out['Frequency'] / (df_out['Recency'] + 1e-5)
    
    # 極端な行動フラグ
    for col in spend_cols:
        df_out[f'is_zero_{col}'] = (df_out[col] == 0).astype(int)
        
    df_out['spend_income_ratio'] = df_out['Monetary'] / (df_out['annual_income'] + 1e-5)
    
    # --- 3. 日付処理 ---
    if 'registration_date' in df_out.columns:
        df_out['registration_date'] = pd.to_datetime(df_out['registration_date'])
        base_date = pd.to_datetime('2026-01-01')
        df_out['days_since_registration'] = (base_date - df_out['registration_date']).dt.days
        df_out.drop(['registration_date'], axis=1, inplace=True)
        
    return df_out

train_feat = create_master_features(train_df, is_train=True)
test_feat = create_master_features(test_df, is_train=False)

# ========================================================
# 2. 教師なし学習（K-Meansクラスタリング）によるペルソナ付与
# ========================================================
# クラスタリングには欠損値処理と標準化が必要
cluster_features = ['age', 'annual_income', 'Monetary', 'Frequency', 'Recency']
train_cluster_data = train_feat[cluster_features].fillna(train_feat[cluster_features].median())
test_cluster_data = test_feat[cluster_features].fillna(train_feat[cluster_features].median())

scaler = StandardScaler()
train_cluster_scaled = scaler.fit_transform(train_cluster_data)
test_cluster_scaled = scaler.transform(test_cluster_data)

# 顧客を5つのペルソナに分類
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
train_feat['customer_cluster'] = kmeans.fit_predict(train_cluster_scaled)
test_feat['customer_cluster'] = kmeans.predict(test_cluster_scaled)

# ========================================================
# 3. 1層目（Level 1）: 3つの強力なモデルの学習とOOF生成
# ========================================================
drop_cols = ['customer_id', 'target']
cat_cols = ['education_level', 'marital_status', 'customer_cluster'] # クラスタもカテゴリ変数として扱う

X = train_feat.drop(columns=[col for col in drop_cols if col in train_feat.columns])
y = train_feat['target']
X_test = test_feat.drop(columns=[col for col in ['customer_id'] if col in test_feat.columns])

X_lgb, X_test_lgb = X.copy(), X_test.copy()
for col in cat_cols:
    X_lgb[col] = X_lgb[col].astype('category')
    X_test_lgb[col] = X_test_lgb[col].astype('category')

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb, oof_xgb, oof_cat = np.zeros(len(X)), np.zeros(len(X)), np.zeros(len(X))
test_lgb, test_xgb, test_cat = np.zeros(len(X_test)), np.zeros(len(X_test)), np.zeros(len(X_test))

print("--- Level 1: ベースモデルの学習（Stackingの準備） ---")

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
    X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
    X_va, y_va = X.iloc[valid_idx], y.iloc[valid_idx]
    X_tr_lgb, X_va_lgb = X_lgb.iloc[train_idx], X_lgb.iloc[valid_idx]
    
    # Target Encoding (XGBoost等のため)
    te = TargetEncoder(cols=cat_cols)
    X_tr_enc = te.fit_transform(X_tr, y_tr)
    X_va_enc = te.transform(X_va)
    X_test_enc = te.transform(X_test)
    
    # --- LightGBM ---
    dtrain_lgb = lgb.Dataset(X_tr_lgb, label=y_tr)
    dvalid_lgb = lgb.Dataset(X_va_lgb, label=y_va, reference=dtrain_lgb)
    lgb_params = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.02, 'max_depth': 6, 'random_state': 42, 'verbosity': -1}
    model_lgb = lgb.train(lgb_params, dtrain_lgb, valid_sets=[dvalid_lgb], num_boost_round=3000, callbacks=[lgb.early_stopping(100, verbose=False)])
    oof_lgb[valid_idx] = model_lgb.predict(X_va_lgb)
    test_lgb += model_lgb.predict(X_test_lgb) / n_splits
    
    # --- XGBoost ---
    xgb_params = {'objective': 'binary:logistic', 'eval_metric': 'auc', 'learning_rate': 0.02, 'max_depth': 5, 'random_state': 42}
    dtrain_xgb, dvalid_xgb, dtest_xgb = xgb.DMatrix(X_tr_enc, label=y_tr), xgb.DMatrix(X_va_enc, label=y_va), xgb.DMatrix(X_test_enc)
    model_xgb = xgb.train(xgb_params, dtrain_xgb, evals=[(dvalid_xgb, 'eval')], num_boost_round=3000, early_stopping_rounds=100, verbose_eval=False)
    oof_xgb[valid_idx] = model_xgb.predict(dvalid_xgb)
    test_xgb += model_xgb.predict(dtest_xgb) / n_splits
    
    # --- CatBoost ---
    model_cat = CatBoostClassifier(iterations=3000, learning_rate=0.02, depth=6, eval_metric='AUC', random_seed=42, verbose=False)
    model_cat.fit(X_tr, y_tr, cat_features=cat_cols, eval_set=(X_va, y_va), early_stopping_rounds=100)
    oof_cat[valid_idx] = model_cat.predict_proba(X_va)[:, 1]
    test_cat += model_cat.predict_proba(X_test)[:, 1] / n_splits
    
    print(f"Fold {fold+1} 完了")

# ========================================================
# 4. 2層目（Level 2）: メタモデルによるスタッキング
# ========================================================
print("\n--- Level 2: ロジスティック回帰によるスタッキング学習 ---")

# 1層目の予測結果を新しい特徴量（データフレーム）としてまとめる
stack_train = pd.DataFrame({'lgb': oof_lgb, 'xgb': oof_xgb, 'cat': oof_cat})
stack_test = pd.DataFrame({'lgb': test_lgb, 'xgb': test_xgb, 'cat': test_cat})

oof_stack = np.zeros(len(X))
test_preds_stack = np.zeros(len(X_test))

# メタモデル（ロジスティック回帰）で再度K-Fold交差検証
for fold, (train_idx, valid_idx) in enumerate(cv.split(stack_train, y)):
    X_tr_stack, y_tr_stack = stack_train.iloc[train_idx], y.iloc[train_idx]
    X_va_stack, y_va_stack = stack_train.iloc[valid_idx], y.iloc[valid_idx]
    
    # メタモデルの学習
    meta_model = LogisticRegression(C=0.1, random_state=42)
    meta_model.fit(X_tr_stack, y_tr_stack)
    
    oof_stack[valid_idx] = meta_model.predict_proba(X_va_stack)[:, 1]
    test_preds_stack += meta_model.predict_proba(stack_test)[:, 1] / n_splits

# ========================================================
# 5. 最終スコアの確認と提出ファイル作成
# ========================================================
print("\n" + "="*50)
print(f"Level 1: LightGBM AUC : {roc_auc_score(y, oof_lgb):.5f}")
print(f"Level 1: XGBoost  AUC : {roc_auc_score(y, oof_xgb):.5f}")
print(f"Level 1: CatBoost AUC : {roc_auc_score(y, oof_cat):.5f}")
print("-" * 50)
print(f"★★★ 最終 Stacking CV AUC: {roc_auc_score(y, oof_stack):.5f} ★★★")
print("="*50)

submission = pd.DataFrame({
    'customer_id': test_df['customer_id'],
    'target': test_preds_stack
})
submission.to_csv("submission_stacking_final.csv", index=False)
print("最強のスタッキングモデルによる予測が完了しました！ `submission_stacking_final.csv` を提出してみてください。")