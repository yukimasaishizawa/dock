import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
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

# 2. 特徴量エンジニアリング (高度な加工)
def engineer_features(df, source_df):
    df_copy = df.copy()
    
    # (A) 欠損値フラグ (Ageの欠損は非常に強いシグナル)
    for col in ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump']:
        df_copy[f'{col}_Missing'] = df_copy[col].isnull().astype(int)
    
    # (B) ポジション別の平均との差 (そのポジション内でどれだけ優れているか)
    # 学習データの統計量を使用して、テストデータにも適用します。
    pos_stats = source_df.groupby('Position')[['Sprint_40yd', 'Weight', 'Vertical_Jump']].mean()
    for col in ['Sprint_40yd', 'Weight', 'Vertical_Jump']:
        df_copy[f'{col}_Pos_Diff'] = df_copy['Position'].map(pos_stats[col]) - df_copy[col]
    
    # (C) 学校の過去実績 (Target Encoding)
    school_rates = source_df.groupby('School')['Drafted'].mean()
    global_mean = source_df['Drafted'].mean()
    df_copy['School_Draft_Rate'] = df_copy['School'].map(school_rates).fillna(global_mean)
    
    # (D) 身体指標 BMI
    df_copy['BMI'] = df_copy['Weight'] / (df_copy['Height'] ** 2)
    
    return df_copy

# 特徴量加工の実行
X_full_eng = engineer_features(train_df, train_df)
X_test_eng = engineer_features(test_df, train_df)

# 使用する列の定義
num_cols = ['Age', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle', 
            'BMI', 'School_Draft_Rate', 'Sprint_40yd_Pos_Diff', 'Weight_Pos_Diff', 'Vertical_Jump_Pos_Diff']
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

# 4. アンサンブルモデルの構築
# 勾配ブースティングとランダムフォレストを組み合わせることで、
# 片方のモデルの弱点をもう片方が補い、予測精度を極限まで高めます。
gb = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=4, random_state=42)
rf = RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42)

# VotingClassifierで2つのモデルの予測結果を統合
ensemble_model = VotingClassifier(estimators=[('gb', gb), ('rf', rf)], voting='soft')

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', ensemble_model)
])

# 5. 内部評価
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
final_pipeline.fit(X_train, y_train)
val_probs = final_pipeline.predict_proba(X_val)[:, 1]
print(f"■ アンサンブルモデルの検証AUCスコア: {roc_auc_score(y_val, val_probs):.4f}")

# 6. 全データで再学習して予測
final_pipeline.fit(X, y)
test_predictions_proba = final_pipeline.predict_proba(X_test)[:, 1]

# 7. 保存
submission = pd.DataFrame({'Id': test_df['Id'], 'Drafted': test_predictions_proba})
submission.to_csv('optimized_high_auc_result.csv', index=False)
print("完了しました！ 'optimized_high_auc_result.csv' に保存されました。")
