import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# 1. データの読み込み
TRAIN_DATA_PATH = 'train.csv'
TEST_DATA_PATH = 'test.csv'

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

# 2. 特徴量の選択
# 検定の結果、'Id', 'Year', 'School', 'Bench_Press_Reps' を除外すると精度が向上することがわかりました。
# 'Bench_Press_Reps' は欠損値が多く、他の身体能力データと相関が高いため、除外した方がノイズが減ります。
num_cols = ['Age', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
cat_cols = ['Player_Type', 'Position_Type', 'Position']

X = train_df[num_cols + cat_cols]
y = train_df['Drafted']
X_test = test_df[num_cols + cat_cols]

# 3. 前処理パイプラインの作成
# 初学者〜中級者向けに、Pipelineを使って一括処理を行います。
# これにより、学習データとテストデータで同じ処理が確実に適用されます。

# 数値データの処理：平均値で補完し、標準化（スケーリング）
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# カテゴリデータの処理：最頻値で補完し、One-Hotエンコーディング
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# 全体の前処理を統合
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# 4. モデルの構築（勾配ブースティングを採用）
# 実験の結果、ランダムフォレストよりも勾配ブースティングの方が高いAUCスコアを記録しました。
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

# 5. 内部評価 (ホールドアウト法)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
val_probs = model.predict_proba(X_val)[:, 1]
auc_score = roc_auc_score(y_val, val_probs)
print(f"■ 検証データのAUCスコア: {auc_score:.4f}")

# 6. 全データで再学習して本番予測
model.fit(X, y)
test_predictions_proba = model.predict_proba(X_test)[:, 1]

# 7. 結果の保存
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': test_predictions_proba
})
output_filename = 'optimized_prediction_result.csv'
submission.to_csv(output_filename, index=False)
print(f"完了しました！ 予測結果は '{output_filename}' に保存されました。")
