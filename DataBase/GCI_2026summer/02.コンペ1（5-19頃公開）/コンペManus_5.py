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
train_df = pd.read_csv('/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/train.csv')
test_df = pd.read_csv('/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/test.csv')
y = train_df['Drafted']

# 2. 特徴量エンジニアリング (AUC 0.9に向けた微調整)
def engineer_features(df, source_df):
    df_copy = df.copy()
    
    # (A) 欠損値のパターンをより詳細に
    missing_cols = ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    for col in missing_cols:
        df_copy[f'{col}_Missing'] = df_copy[col].isnull().astype(int)
    df_copy['Missing_Score'] = df_copy[missing_cols].isnull().sum(axis=1)
    
    # (B) スピードスコア (NFLで重視される「体重を加味した速さ」)
    # 体重(kg) / 40ヤード走(秒)の4乗
    df_copy['Speed_Weight_Score'] = df_copy['Weight'] / (df_copy['Sprint_40yd']**4)
    
    # (C) ポジション別Zスコア (そのポジションの平均からどれだけ逸脱しているか)
    pos_stats = source_df.groupby('Position')[['Sprint_40yd', 'Weight', 'Vertical_Jump']].agg(['mean', 'std'])
    for col in ['Sprint_40yd', 'Weight', 'Vertical_Jump']:
        mean = df_copy['Position'].map(pos_stats[col]['mean'])
        std = df_copy['Position'].map(pos_stats[col]['std']).replace(0, 1)
        df_copy[f'{col}_Pos_Z'] = (df_copy[col] - mean) / std
    
    # (D) 学校の指名力 (Smoothingを調整してTestデータに合わせる)
    global_mean = source_df['Drafted'].mean()
    m = 5 # Smoothing: 低めに設定して個別の学校の特徴を強める
    school_stats = source_df.groupby('School')['Drafted'].agg(['count', 'mean'])
    school_encoded = (school_stats['count'] * school_stats['mean'] + m * global_mean) / (school_stats['count'] + m)
    df_copy['School_Power'] = df_copy['School'].map(school_encoded).fillna(global_mean)
    
    return df_copy

# 加工実行
X_full_eng = engineer_features(train_df, train_df)
X_test_eng = engineer_features(test_df, train_df)

# 列定義
num_cols = ['Age', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle', 
            'Speed_Weight_Score', 'School_Power', 'Missing_Score', 'Sprint_40yd_Pos_Z', 'Weight_Pos_Z', 'Vertical_Jump_Pos_Z']
num_cols += [f'{col}_Missing' for col in ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']]
cat_cols = ['Player_Type', 'Position_Type', 'Position']

X = X_full_eng[num_cols + cat_cols]
X_test = X_test_eng[num_cols + cat_cols]

# 3. 前処理
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
])

# 4. スタッキング・アンサンブルの微調整
# 学習率を下げて、より多くの決定木(n_estimators)でじっくり学習させます。
base_models = [
    ('gb', GradientBoostingClassifier(n_estimators=500, learning_rate=0.02, max_depth=4, random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=12, random_state=42))
]

# メタモデルに正則化(C=0.1)を加えて過学習を抑制
stacking_model = StackingClassifier(
    estimators=base_models, 
    final_estimator=LogisticRegression(C=0.1),
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
print(f"■ 最終微調整モデルの検証AUCスコア: {roc_auc_score(y_val, val_probs):.4f}")

# 6. 本番予測 (Testデータの傾向を考慮)
# 全データで学習
final_pipeline.fit(X, y)
test_predictions_proba = final_pipeline.predict_proba(X_test)[:, 1]

# 7. 保存
submission = pd.DataFrame({'Id': test_df['Id'], 'Drafted': test_predictions_proba})
submission.to_csv('optimized_0.9_attempt_result.csv', index=False)
print("完了しました！ 'optimized_0.9_attempt_result.csv' に保存されました。")
