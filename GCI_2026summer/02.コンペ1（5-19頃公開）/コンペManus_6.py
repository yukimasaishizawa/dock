import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# 1. データの読み込み
train_df = pd.read_csv('/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/train.csv')
test_df = pd.read_csv('/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/test.csv')
y_train = train_df['Drafted']

# 2. 特徴量エンジニアリング (既存のデータを組み合わせて新しい指標を作成)
def engineer_features(df, source_df):
    df_copy = df.copy()
    
    # (A) 身体能力の複合指標 (Composite Athletic Scores)
    # スピードスコア: 体重と40ヤード走の組み合わせ (NFLスカウトが重視)
    df_copy['Speed_Score'] = (df_copy['Weight'] * 2.20462) / (df_copy['Sprint_40yd'] ** 2)
    
    # 瞬発力指標: 垂直跳びと立ち幅跳びの組み合わせ
    df_copy['Explosiveness'] = df_copy['Vertical_Jump'] * df_copy['Broad_Jump']
    
    # (B) ポジション内での相対評価 (Position-Relative Rank)
    # 同じポジションの中で、40ヤード走が何パーセンタイルに位置するか
    df_copy['Sprint_Pos_Rank'] = df_copy.groupby('Position')['Sprint_40yd'].rank(pct=True)
    
    # (C) 欠損値の戦略的フラグ (Strategic Missingness)
    # Ageが欠損していることは、指名において非常に重要なシグナル
    df_copy['Age_Missing'] = df_copy['Age'].isnull().astype(int)
    # 主要な身体測定を一つでも欠いているか
    df_copy['Any_Combine_Missing'] = df_copy[['Vertical_Jump', 'Broad_Jump', 'Agility_3cone']].isnull().any(axis=1).astype(int)
    
    # (D) 学校の指名実績 (K-Fold的な平滑化を模した処理)
    global_mean = source_df['Drafted'].mean()
    m = 15 # Smoothing: 高めに設定してテストデータへの汎化性能を高める
    school_stats = source_df.groupby('School')['Drafted'].agg(['count', 'mean'])
    school_encoded = (school_stats['count'] * school_stats['mean'] + m * global_mean) / (school_stats['count'] + m)
    df_copy['School_Rate'] = df_copy['School'].map(school_encoded).fillna(global_mean)
    
    return df_copy

# 特徴量加工の実行
train_eng = engineer_features(train_df, train_df)
test_eng = engineer_features(test_df, train_df)

# カテゴリ変数の処理 (HistGradientBoostingはカテゴリ型を直接扱える)
cat_cols = ['Player_Type', 'Position_Type', 'Position']
for col in cat_cols:
    train_eng[col] = train_eng[col].astype('category')
    test_eng[col] = test_eng[col].astype('category')

# 使用する特徴量の選択
features = [
    'Age', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 
    'Broad_Jump', 'Agility_3cone', 'Shuttle', 'Speed_Score', 'Explosiveness', 
    'Sprint_Pos_Rank', 'Age_Missing', 'Any_Combine_Missing', 'School_Rate'
] + cat_cols

X = train_eng[features]
X_test = test_eng[features]

# 3. モデルの構築 (過学習を抑え、汎化性能を最大化した設定)
# HistGradientBoostingClassifier は欠損値を自動で処理し、非常に強力です。
model = HistGradientBoostingClassifier(
    max_iter=400,            # 学習回数
    learning_rate=0.02,      # 学習率（低めに設定してじっくり学習）
    max_depth=5,             # 木の深さ（深すぎると過学習するため制限）
    l2_regularization=15.0,  # L2正則化（過学習を強力に抑制）
    categorical_features=[features.index(c) for c in cat_cols], # カテゴリ変数の指定
    random_state=42
)

# 4. 内部評価 (交差検証)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in cv.split(X, y_train):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    model.fit(X_tr, y_tr)
    probs = model.predict_proba(X_val)[:, 1]
    cv_scores.append(roc_auc_score(y_val, probs))

print(f"■ 堅牢化モデルの交差検証AUCスコア: {np.mean(cv_scores):.4f}")

# 5. 本番予測
model.fit(X, y_train)
test_predictions_proba = model.predict_proba(X_test)[:, 1]

# 6. 結果の保存
submission = pd.DataFrame({
    'Id': test_df['Id'],
    'Drafted': test_predictions_proba
})
submission.to_csv('final_robust_prediction.csv', index=False)
print("完了しました！ 'final_robust_prediction.csv' に予測結果が保存されました。")
