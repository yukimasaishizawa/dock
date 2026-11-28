import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
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

# 2. 高度な特徴量エンジニアリング
def engineer_features(df, source_df):
    df_copy = df.copy()
    
    # (A) 欠損値シグナルの抽出
    missing_cols = ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    for col in missing_cols:
        df_copy[f'{col}_Missing'] = df_copy[col].isnull().astype(int)
    
    # 合計欠損数
    df_copy['Total_Missing_Count'] = df_copy[missing_cols].isnull().sum(axis=1)
    
    # (B) ポジション別相対評価 (Z-Score)
    pos_stats = source_df.groupby('Position')[['Sprint_40yd', 'Weight', 'Vertical_Jump']].agg(['mean', 'std'])
    for col in ['Sprint_40yd', 'Weight', 'Vertical_Jump']:
        mean = df_copy['Position'].map(pos_stats[col]['mean'])
        std = df_copy['Position'].map(pos_stats[col]['std']).replace(0, 1)
        df_copy[f'{col}_Pos_Z'] = (df_copy[col] - mean) / std
    
    # (C) 学校実績の平滑化ターゲットエンコーディング
    global_mean = source_df['Drafted'].mean()
    m = 10
    school_stats = source_df.groupby('School')['Drafted'].agg(['count', 'mean'])
    school_encoded = (school_stats['count'] * school_stats['mean'] + m * global_mean) / (school_stats['count'] + m)
    df_copy['School_Success_Rate'] = df_copy['School'].map(school_encoded).fillna(global_mean)
    
    # (D) 身体指標 BMI
    df_copy['BMI'] = df_copy['Weight'] / (df_copy['Height'] ** 2)
    
    return df_copy

# 特徴量加工
X_full_eng = engineer_features(train_df, train_df)
X_test_eng = engineer_features(test_df, train_df)

# 使用する列の定義 (重複を排除)
num_cols = ['Age', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle', 
            'BMI', 'School_Success_Rate', 'Total_Missing_Count', 'Sprint_40yd_Pos_Z', 'Weight_Pos_Z', 'Vertical_Jump_Pos_Z']
num_cols += [f'{col}_Missing' for col in ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']]
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

# 4. スタッキング・アンサンブル
base_models = [
    ('gb', GradientBoostingClassifier(n_estimators=300, learning_rate=0.03, max_depth=4, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=300, max_depth=10, random_state=42))
]

stacking_model = StackingClassifier(
    estimators=base_models, 
    final_estimator=LogisticRegression(),
    cv=5
)

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', stacking_model)
])

# 5. 内部評価
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
final_pipeline.fit(X_train, y_train)
val_probs = final_pipeline.predict_proba(X_val)[:, 1]
print(f"■ スタッキングモデルの検証AUCスコア: {roc_auc_score(y_val, val_probs):.4f}")

# 6. 全データで再学習して本番予測
final_pipeline.fit(X, y)
test_predictions_proba = final_pipeline.predict_proba(X_test)[:, 1]

# 7. 保存
submission = pd.DataFrame({'Id': test_df['Id'], 'Drafted': test_predictions_proba})
submission.to_csv('optimized_high_performance_result.csv', index=False)
print("完了しました！ 'optimized_high_performance_result.csv' に保存されました。")
