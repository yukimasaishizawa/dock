import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

# 1. データの読み込み
df_mean = pd.read_csv("train_mean.csv")

# 2. 学習データの前処理（ダミー変数化）
edu = pd.get_dummies(df_mean["education_level"], dtype=int)
mari = pd.get_dummies(df_mean["marital_status"], dtype=int)
df_mean = pd.concat([df_mean, edu, mari], axis=1)

df_mean["spend_sweets"] = df_mean["spend_sweets"].astype(int)
df_mean["spend_gold"] = df_mean["spend_gold"].astype(int)

# モデルに使う28個の特徴量リスト
features = [
    'Single' ,'Married', 'Partner' ,'Widowed', 'Divorced', 'Other',
    'Bachelor' ,'Doctorate', 'Master', 'Associate', 'High School',
    "spend_sweets","spend_gold","birth_year","annual_income",
    "num_children","num_teenagers","days_since_last_purchase",
    "spend_wines","spend_fruits","spend_meat","spend_fish",
    "deals_purchases","web_purchases","catalog_purchases",
    "store_purchases","monthly_web_visits","has_complaint"
]

x_train = df_mean[features]
y_train = df_mean["target"]

# 3. 標準化（学習データでfit）
sc = StandardScaler()
x_train_scaled = sc.fit_transform(x_train)

# 4. モデルの学習
model_mean = LinearRegression()
model_mean.fit(x_train_scaled, y_train)

# ========================================================
# 5. テストデータの読み込みと【欠損値・ダミー変数対策】
# ========================================================
df_new = pd.read_csv("test.csv")

# テストデータ独自の数値欠損値を、学習データの平均値などで穴埋め
# （特に annual_income などに欠損があることが多いです）
df_new = df_new.fillna(df_mean.mean(numeric_only=True))

# テストデータもダミー変数化
edu_new = pd.get_dummies(df_new["education_level"], dtype=int)
mari_new = pd.get_dummies(df_new["marital_status"], dtype=int)
df_new_processed = pd.concat([df_new, edu_new, mari_new], axis=1)

# 型変換
df_new_processed["spend_sweets"] = df_new_processed["spend_sweets"].fillna(0).astype(int)
df_new_processed["spend_gold"] = df_new_processed["spend_gold"].fillna(0).astype(int)

# 学習時と同じ特徴量を抽出（テスト側に存在しないダミー変数列はここで NaN になります）
x_test = df_new_processed.reindex(columns=features)

# 【重要】存在しなくて NaN になったダミー変数列や、その他漏れた NaN を一括で 0 に置換
x_test = x_test.fillna(0)

# テストデータを標準化
x_test_scaled = sc.transform(x_test)

# 6. 予測の実行
predictions = model_mean.predict(x_test_scaled)

# 7. 提出用ファイルの作成
df_submission = pd.DataFrame({
    "customer_id": df_new["customer_id"],
    "target": predictions
})

# CSVとして出力
df_submission.to_csv("submission_linear.csv", index=False)
print("エラーなく予測が完了し、submission.csv を保存しました！")