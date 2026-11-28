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
TRAIN_DATA_PATH = '/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/train.csv'
TEST_DATA_PATH = '/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/test.csv'

train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

# 2. 特徴量エンジニアリング (精度の鍵となる加工)
def engineer_features(df, source_df):
    df_copy = df.copy()
    
    # (A) 身体指標の作成: BMI
    df_copy['BMI'] = df_copy['Weight'] / (df_copy['Height'] ** 2)
    
    # (B) 欠損値フラグの作成 (非常に重要: 欠損があること自体が「指名されない」強力なシグナル)
    # 特に Age の欠損は指名確率に大きく関わります。
    cols_to_check = ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    for col in cols_to_check:
        df_copy[f'{col}_Missing'] = df_copy[col].isnull().astype(int)
    
    # (C) 学校の過去実績 (Target Encoding)
    # その学校から過去に何パーセント指名されたかを特徴量にします。
    school_rates = source_df.groupby('School')['Drafted'].mean()
    global_mean = source_df['Drafted'].mean()
    df_copy['School_Draft_Rate'] = df_copy['School'].map(school_rates).fillna(global_mean)
    
    return df_copy

# 特徴量加工の実行
X_full_eng = engineer_features(train_df, train_df)
X_test_eng = engineer_features(test_df, train_df)

# 使用する列の定義
num_cols = ['Age', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle', 'BMI', 'School_Draft_Rate']
# 欠損フラグ列を追加
num_cols += [c for c in X_full_eng.columns if '_Missing' in c]
cat_cols = ['Player_Type', 'Position_Type', 'Position']

y = train_df['Drafted']
X = X_full_eng[num_cols + cat_cols]
X_test = X_test_eng[num_cols + cat_cols]

# 3. 前処理パイプライン
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols),
        ('cat', categorical_transformer, cat_cols)
    ])

# 4. モデル構築 (勾配ブースティング)
# 推定器の数を増やし、学習率を調整して精度を高めます。
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42))
])

# 5. 内部評価
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
val_probs = model.predict_proba(X_val)[:, 1]
print(f"■ 検証データのAUCスコア: {roc_auc_score(y_val, val_probs):.4f}")

# 6. 全データで再学習して予測
model.fit(X, y)
test_predictions_proba = model.predict_proba(X_test)[:, 1]

# 7. 保存
submission = pd.DataFrame({'Id': test_df['Id'], 'Drafted': test_predictions_proba})
submission.to_csv('optimized_0.85_result.csv', index=False)
print("完了しました！ 'optimized_0.85_result.csv' に保存されました。")
