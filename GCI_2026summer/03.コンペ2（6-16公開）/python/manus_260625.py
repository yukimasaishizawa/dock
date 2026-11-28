import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

def preprocess_data(df):
    df = df.copy()
    
    # 1. Date Features
    df['registration_date'] = pd.to_datetime(df['registration_date'])
    ref_date = pd.to_datetime('2014-12-31')
    df['days_reg'] = (ref_date - df['registration_date']).dt.days
    df.drop('registration_date', axis=1, inplace=True)
    
    # 2. Demographic & Household
    df['age'] = 2014 - df['birth_year']
    df['total_children'] = df['num_children'] + df['num_teenagers']
    df['is_parent'] = (df['total_children'] > 0).astype(int)
    
    # 3. Financial & Spending
    df['annual_income'] = df['annual_income'].fillna(df['annual_income'].median())
    spend_cols = [col for col in df.columns if col.startswith('spend_')]
    df['total_spend'] = df[spend_cols].sum(axis=1)
    df['income_per_spend'] = df['annual_income'] / (df['total_spend'] + 1)
    
    # 4. Purchases & Channels
    purchase_cols = [col for col in df.columns if 'purchases' in col]
    df['total_purchases'] = df[purchase_cols].sum(axis=1)
    
    # 5. Ratios
    for col in spend_cols:
        df[f'{col}_ratio'] = df[col] / (df['total_spend'] + 1)
    for col in purchase_cols:
        df[f'{col}_ratio'] = df[col] / (df['total_purchases'] + 1)
        
    # 6. Interaction Features
    df['meat_wine_ratio'] = df['spend_meat'] / (df['spend_wines'] + 1)
    df['store_web_ratio'] = df['store_purchases'] / (df['web_purchases'] + 1)
    
    # 7. Encoding
    edu_map = {'High School': 0, 'Associate': 1, 'Bachelor': 2, 'Master': 3, 'Doctorate': 4}
    df['education_level'] = df['education_level'].map(edu_map)
    
    # Marital status - simplified grouping
    df['is_single'] = df['marital_status'].isin(['Single', 'Divorced', 'Widowed']).astype(int)
    df = pd.get_dummies(df, columns=['marital_status'], prefix='marital')
    
    if 'customer_id' in df.columns:
        df.drop('customer_id', axis=1, inplace=True)
        
    return df

# データの読み込み
# ※ ご自身の環境に合わせてパスを適宜変更してください
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 提出ファイル用にテストデータの customer_id を保存しておく
test_id = test['customer_id'].copy()

# 前処理の適用
train_df = preprocess_data(train)
test_df = preprocess_data(test)

X = train_df.drop('target', axis=1)
y = train_df['target']

# テストデータの列順を訓練データ(X)と完全に一致させる
test_df = test_df[X.columns]

# Use 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
auc_scores = []

# 各フォールドの予測値を格納する配列（ゼロ初期化）
test_preds = np.zeros(len(test_df))

# Conservative parameters to prevent overfitting while allowing the model to learn
lgb_params = {
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42,
    'learning_rate': 0.002,
    'num_leaves': 15,
    'max_depth': 4,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.6,
    'bagging_freq': 5,
    'lambda_l1': 10.0,
    'lambda_l2': 10.0,
    'min_child_samples': 50,
    'is_unbalance': True
}

print("Starting Cross-Validation v7...")
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    dtrain = lgb.Dataset(X_train, label=y_train)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)
    
    model = lgb.train(
        lgb_params,
        dtrain,
        valid_sets=[dtrain, dval],
        valid_names=['train', 'valid'],
        num_boost_round=20000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=1000),
            lgb.log_evaluation(period=0)
        ]
    )
    
    # 制御用検証データの予測
    y_pred = model.predict(X_val, num_iteration=model.best_iteration)
    auc = roc_auc_score(y_val, y_pred)
    auc_scores.append(auc)
    print(f"Fold {fold+1} AUC: {auc:.4f}")
    
    # テストデータの予測（各フォールドの予測値を足し合わせて、後で5で割る）
    test_preds += model.predict(test_df, num_iteration=model.best_iteration) / skf.n_splits

mean_auc = np.mean(auc_scores)
print(f"\nMean AUC: {mean_auc:.4f}")
print(f"AUC Std: {np.std(auc_scores):.4f}")

# 提出用ファイルの作成
submission = pd.DataFrame({
    'customer_id': test_id,
    'target': test_preds
})

# 指定されたファイル形式で書き出し (インデックスは含めない)
submission.to_csv('submission_manus.csv', index=False)
print("\n提出用ファイル 'submission_manus.csv' を書き出しました。")
print(submission.head())