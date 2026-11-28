import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ========================================================
# 1. データの読み込みと【究極の特徴量エンジニアリング】
# ========================================================
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

def create_killer_features(df):
    df_out = df.copy()
    
    # --- A. 外れ値の処理（ノイズ除去） ---
    # 常識的な範囲にクリッピング（上限・下限で丸める）
    df_out['birth_year'] = df_out['birth_year'].clip(lower=1940)
    # 年収の欠損値を「学歴の中央値」で埋めた上でクリッピング
    df_out['annual_income'] = df_out.groupby('education_level')['annual_income'].transform(lambda x: x.fillna(x.median()))
    df_out['annual_income'] = df_out['annual_income'].clip(upper=150000)
    
    # --- B. 基本属性とクロス特徴量 ---
    df_out['age'] = 2026 - df_out['birth_year']
    df_out['total_children'] = df_out['num_children'] + df_out['num_teenagers']
    # カテゴリの掛け合わせ（例: "Bachelor_Single"）
    df_out['edu_marital'] = df_out['education_level'].astype(str) + "_" + df_out['marital_status'].astype(str)
    
    # --- C. 消費・購買の集計 ---
    spend_cols = ['spend_wines', 'spend_fruits', 'spend_meat', 'spend_fish', 'spend_sweets', 'spend_gold']
    purchase_cols = ['deals_purchases', 'web_purchases', 'catalog_purchases', 'store_purchases']
    df_out['total_spend'] = df_out[spend_cols].sum(axis=1)
    df_out['total_purchases'] = df_out[purchase_cols].sum(axis=1)
    
    # --- D. 【重要】行動心理を表すキラーレシオ ---
    # 1. ディスカウントハンター度（割引購入の割合）
    df_out['discount_seeker_ratio'] = df_out['deals_purchases'] / (df_out['total_purchases'] + 1e-5)
    # 2. Webコンバージョン率（訪問したのに買わない人をあぶり出す）
    df_out['web_cv_rate'] = df_out['web_purchases'] / (df_out['monthly_web_visits'] + 1e-5)
    # 3. エンゲル係数・スーパー依存度
    df_out['spend_income_ratio'] = df_out['total_spend'] / (df_out['annual_income'] + 1e-5)
    # 4. ワイン・肉への執着度（高単価商品）
    df_out['wine_ratio'] = df_out['spend_wines'] / (df_out['total_spend'] + 1e-5)
    df_out['meat_ratio'] = df_out['spend_meat'] / (df_out['total_spend'] + 1e-5)
    
    # --- E. 日付・コホート特徴量 ---
    if 'registration_date' in df_out.columns:
        df_out['registration_date'] = pd.to_datetime(df_out['registration_date'])
        # 単純な経過日数
        base_date = pd.to_datetime('2026-01-01')
        df_out['days_since_registration'] = (base_date - df_out['registration_date']).dt.days
        # コホート（登録年、登録月）をカテゴリとして抽出
        df_out['reg_year'] = df_out['registration_date'].dt.year.astype(str)
        df_out['reg_month'] = df_out['registration_date'].dt.month.astype(str)
        df_out.drop(['registration_date'], axis=1, inplace=True)
        
    return df_out

train_feat = create_killer_features(train_df)
test_feat = create_killer_features(test_df)

# ========================================================
# 2. 学習準備（カテゴリ変数の指定）
# ========================================================
drop_cols = ['customer_id', 'target']
# CatBoost用に文字列のまま扱うカテゴリ列
cat_cols = ['education_level', 'marital_status', 'edu_marital', 'reg_year', 'reg_month']

X = train_feat.drop(columns=[col for col in drop_cols if col in train_feat.columns])
y = train_feat['target']
X_test = test_feat.drop(columns=[col for col in ['customer_id'] if col in test_feat.columns])

# LightGBM用にCategory型へ変換
X_lgb = X.copy()
X_test_lgb = X_test.copy()
for col in cat_cols:
    if col in X_lgb.columns:
        X_lgb[col] = X_lgb[col].astype('category')
        X_test_lgb[col] = X_test_lgb[col].astype('category')

# 欠損値の最終処理（CatBoostは数値のNaNを嫌う場合があるため）
X = X.fillna(-999)
X_test = X_test.fillna(-999)

n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

oof_lgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))
test_preds_lgb = np.zeros(len(X_test))
test_preds_cat = np.zeros(len(X_test))

# ========================================================
# 3. モデル学習（LightGBM & CatBoost）
# ========================================================
print("--- キラー特徴量を用いた学習を開始します ---")

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
    # LightGBM
    X_tr_lgb, y_tr = X_lgb.iloc[train_idx], y.iloc[train_idx]
    X_va_lgb, y_va = X_lgb.iloc[valid_idx], y.iloc[valid_idx]
    
    dtrain_lgb = lgb.Dataset(X_tr_lgb, label=y_tr)
    dvalid_lgb = lgb.Dataset(X_va_lgb, label=y_va, reference=dtrain_lgb)
    
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.02, 
        'max_depth': 6, 'num_leaves': 31, 'min_child_samples': 20,
        'colsample_bytree': 0.7, 'subsample': 0.8, 
        'random_state': 42 + fold, 'verbosity': -1
    }
    model_lgb = lgb.train(
        lgb_params, dtrain_lgb, valid_sets=[dvalid_lgb],
        num_boost_round=3000, callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    oof_lgb[valid_idx] = model_lgb.predict(X_va_lgb)
    test_preds_lgb += model_lgb.predict(X_test_lgb) / n_splits

    # CatBoost
    X_tr_cat, X_va_cat = X.iloc[train_idx], X.iloc[valid_idx]
    model_cat = CatBoostClassifier(
        iterations=3000, learning_rate=0.02, depth=6,
        eval_metric='AUC', random_seed=42 + fold, verbose=False
    )
    model_cat.fit(
        X_tr_cat, y_tr, cat_features=[c for c in cat_cols if c in X_tr_cat.columns],
        eval_set=(X_va_cat, y_va), early_stopping_rounds=100
    )
    oof_cat[valid_idx] = model_cat.predict_proba(X_va_cat)[:, 1]
    test_preds_cat += model_cat.predict_proba(X_test)[:, 1] / n_splits

    print(f"Fold {fold+1} 完了")

# ========================================================
# 4. スコア確認と提出ファイル
# ========================================================
print("\n" + "="*50)
print(f"LightGBM 単体 AUC: {roc_auc_score(y, oof_lgb):.5f}")
print(f"CatBoost 単体 AUC: {roc_auc_score(y, oof_cat):.5f}")

# 50:50 の単純平均ブレンド
oof_blend = (oof_lgb + oof_cat) / 2.0
final_auc = roc_auc_score(y, oof_blend)

print("-" * 50)
print(f"★★★ 最終 Killer Features AUC: {final_auc:.5f} ★★★")
print("=" * 50)

final_preds = (test_preds_lgb + test_preds_cat) / 2.0
submission = pd.DataFrame({
    'customer_id': test_df['customer_id'],
    'target': final_preds
})
submission.to_csv("submission_killer_features.csv", index=False)
print("予測完了！ `submission_killer_features.csv` が保存されました。")
