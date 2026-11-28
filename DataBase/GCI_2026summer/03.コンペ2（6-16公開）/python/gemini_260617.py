import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# ========================================================
# 1. データの読み込みと特徴量エンジニアリング
# ========================================================
# READMEの定義通り、元のtrain.csvとtest.csvを使用することを想定
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

def create_features(df):
    """
    ドメイン知識に基づく常識的な特徴量エンジニアリングを実行
    """
    df_out = df.copy()
    
    # --- A. 顧客の基本属性 ---
    # 年齢（2026年時点と仮定して算出）
    df_out['age'] = 2026 - df_out['birth_year']
    
    # 子供の総数
    df_out['total_children'] = df_out['num_children'] + df_out['num_teenagers']
    
    # 「親であるか」のフラグ（子供が1人以上いれば1、いなければ0）
    df_out['is_parent'] = (df_out['total_children'] > 0).astype(int)

    # --- B. 購買金額（Spend）の特徴量 ---
    spend_cols = ['spend_wines', 'spend_fruits', 'spend_meat', 'spend_fish', 'spend_sweets', 'spend_gold']
    # 各カテゴリの総消費額
    df_out['total_spend'] = df_out[spend_cols].sum(axis=1)
    
    # 総消費額に対するワインや肉（高単価）の割合（ゼロ除算回避のため分母に1e-5を足す）
    df_out['wine_ratio'] = df_out['spend_wines'] / (df_out['total_spend'] + 1e-5)
    df_out['meat_ratio'] = df_out['spend_meat'] / (df_out['total_spend'] + 1e-5)

    # --- C. 購買チャネルの特徴量 ---
    purchase_cols = ['deals_purchases', 'web_purchases', 'catalog_purchases', 'store_purchases']
    # 総購入回数
    df_out['total_purchases'] = df_out[purchase_cols].sum(axis=1)
    
    # 1回あたりの平均購入額
    df_out['avg_spend_per_purchase'] = df_out['total_spend'] / (df_out['total_purchases'] + 1e-5)

    # --- D. 日付関連 ---
    if 'registration_date' in df_out.columns:
        df_out['registration_date'] = pd.to_datetime(df_out['registration_date'])
        # 基準日（学習データの最新日などを設定）から何日経過しているか
        base_date = pd.to_datetime('2026-01-01') # コンペの背景に合わせて適宜変更してください
        df_out['days_since_registration'] = (base_date - df_out['registration_date']).dt.days
        df_out.drop(['registration_date'], axis=1, inplace=True)
        
    return df_out

# 特徴量の生成
train_feat = create_features(train_df)
test_feat = create_features(test_df)

# カテゴリ変数の処理（LightGBM用にCategory型へ変換）
cat_cols = ['education_level', 'marital_status']
for col in cat_cols:
    train_feat[col] = train_feat[col].astype('category')
    test_feat[col] = test_feat[col].astype('category')

# 説明変数と目的変数に分割（ID等の不要なカラムを除外）
drop_cols = ['customer_id', 'target']
X = train_feat.drop(columns=[col for col in drop_cols if col in train_feat.columns])
y = train_feat['target']
X_test = test_feat.drop(columns=[col for col in ['customer_id'] if col in test_feat.columns])

# ========================================================
# 2. LightGBMによる交差検証（Cross Validation）学習
# ========================================================
n_splits = 5
cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

# OOF (Out-of-Fold) 予測を格納する配列
oof_preds = np.zeros(len(X))
# テストデータの予測を格納する配列
test_preds = np.zeros(len(X_test))

# 過学習を防ぐためのハイパーパラメータ設定
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.05,    # 学習率を下げてじっくり学習
    'max_depth': 6,           # 木の深さを制限して過学習を防ぐ
    'num_leaves': 31,
    'min_child_samples': 20,  # 葉に落ちる最小データ数
    'subsample': 0.8,         # データのサンプリング割合
    'colsample_bytree': 0.8,  # 特徴量のサンプリング割合
    'random_state': 42,
    'importance_type': 'gain'
}

print("--- LightGBM K-Fold Cross Validation を開始します ---")

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    
    # LightGBM用のデータセットを作成
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)
    
    # 学習 (early_stoppingを使用して過学習を防ぐ)
    model = lgb.train(
        params=lgb_params,
        train_set=dtrain,
        num_boost_round=10000,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0) # 出力を抑えるため0
        ]
    )
    
    # バリデーションデータへの予測
    valid_preds = model.predict(X_valid, num_iteration=model.best_iteration)
    oof_preds[valid_idx] = valid_preds
    
    # 各FoldのAUCを計算
    fold_auc = roc_auc_score(y_valid, valid_preds)
    print(f"Fold {fold+1}: AUC = {fold_auc:.5f} (Best Iteration: {model.best_iteration})")
    
    # テストデータへの予測（各Foldの予測の平均をとるアンサンブル）
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / n_splits

# 全体でのOOFスコア（これが手元での信頼できるAUCの評価値です）
cv_auc = roc_auc_score(y, oof_preds)
print("-" * 50)
print(f"★ CV Overall AUC: {cv_auc:.5f}")
print("-" * 50)

# ========================================================
# 3. 提出ファイルの作成
# ========================================================
submission = pd.DataFrame({
    'customer_id': test_df['customer_id'],
    'target': test_preds
})

submission.to_csv("submission_lgbm.csv", index=False)
print("予測が完了しました！ `submission_lgbm.csv` を保存しました。")