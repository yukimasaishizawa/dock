import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ========================================================
# 1. 各データセットの読み込みと評価
# ========================================================

# 比較するデータセットのファイル名辞書
datasets = {
    "Mean (平均値補完)": "train_mean.csv",
    "Median (中央値補完)": "train_median.csv",
    "Mode (最頻値補完)": "train_mode.csv"
}

# モデルに使う28個の特徴量リスト
features = [
    'Single', 'Married', 'Partner', 'Widowed', 'Divorced', 'Other',
    'Bachelor', 'Doctorate', 'Master', 'Associate', 'High School',
    "spend_sweets", "spend_gold", "birth_year", "annual_income",
    "num_children", "num_teenagers", "days_since_last_purchase",
    "spend_wines", "spend_fruits", "spend_meat", "spend_fish",
    "deals_purchases", "web_purchases", "catalog_purchases",
    "store_purchases", "monthly_web_visits", "has_complaint"
]

# 各データセットの結果を記録する辞書
results = {}
trained_models = {}
scalers = {}
train_dfs = {}

print("--- 各データセットの交差検証（AUCスコア）を開始します ---")

for name, file_path in datasets.items():
    # データの読み込み
    df = pd.read_csv(file_path)
    
    # 前処理（ダミー変数化）
    edu = pd.get_dummies(df["education_level"], dtype=int)
    mari = pd.get_dummies(df["marital_status"], dtype=int)
    df_processed = pd.concat([df, edu, mari], axis=1)
    
    # 型変換
    df_processed["spend_sweets"] = df_processed["spend_sweets"].astype(int)
    df_processed["spend_gold"] = df_processed["spend_gold"].astype(int)
    
    # 特徴量とターゲットの分離
    x_train = df_processed[features]
    y_train = df_processed["target"].astype(int)
    
    # 標準化
    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    
    # ロジスティック回帰モデルの定義
    model = LogisticRegression(max_iter=1000, random_state=42)
    
    # 5分割の交差検証でROC-AUCスコアを算出（StratifiedKFoldでラベルの比率を維持）
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, x_train_scaled, y_train, cv=cv, scoring='roc_auc')
    mean_auc = np.mean(auc_scores)
    
    # 記録
    results[name] = mean_auc
    print(f"{name}: 平均AUC = {mean_auc:.5f} (各Fold: {', '.join([f'{s:.4f}' for s in auc_scores])})")
    
    # 全データで再学習させて、本番予測用に保存
    model.fit(x_train_scaled, y_train)
    trained_models[name] = model
    scalers[name] = sc
    train_dfs[name] = df  # テストデータの欠損値補完用に元データを保持

print("-" * 50)

# ========================================================
# 2. 最もAUCスコアが高いモデルの選定
# ========================================================
best_dataset_name = max(results, key=results.get)
best_auc = results[best_dataset_name]

print(f"★ 最も高いスコア: {best_dataset_name} (AUC: {best_auc:.5f})")
print(f"-> このモデルを使用してテストデータの予測を行います。")
print("-" * 50)

# 最適なモデル、スケーラー、学習用データを抽出
best_model = trained_models[best_dataset_name]
best_scaler = scalers[best_dataset_name]
best_train_df = train_dfs[best_dataset_name]

# ========================================================
# 3. テストデータの読み込みと【最適な設定】での前処理
# ========================================================
df_new = pd.read_csv("test.csv")

# 最も成績の良かった学習データの数値（平均値や中央値など）をベースに、テストデータの欠損値を穴埋め
df_new = df_new.fillna(best_train_df.mean(numeric_only=True))

# テストデータもダミー変数化
edu_new = pd.get_dummies(df_new["education_level"], dtype=int)
mari_new = pd.get_dummies(df_new["marital_status"], dtype=int)
df_new_processed = pd.concat([df_new, edu_new, mari_new], axis=1)

df_new_processed["spend_sweets"] = df_new_processed["spend_sweets"].fillna(0).astype(int)
df_new_processed["spend_gold"] = df_new_processed["spend_gold"].fillna(0).astype(int)

# 列を学習時と統一
x_test = df_new_processed.reindex(columns=features).fillna(0)

# 最適なスケーラーで標準化
x_test_scaled = best_scaler.transform(x_test)

# ========================================================
# 4. 予測と提出ファイルの作成
# ========================================================
# 確率を出力
predictions = best_model.predict_proba(x_test_scaled)[:, 1]

# 提出用DataFrame
df_submission = pd.DataFrame({
    "customer_id": df_new["customer_id"],
    "target": predictions
})

# 出力
df_submission.to_csv("submission.csv", index=False)
print(f"予測が完了しました！ `{best_dataset_name}` のモデルを使用した `submission.csv` を保存しました。")