import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier

# Load data
train = pd.read_csv('/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/train.csv')
test = pd.read_csv('/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/test.csv')

def engineer_features(df):
    df = df.copy()
    
    # Missing value indicators
    for col in ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']:
        df[f'{col}_is_missing'] = df[col].isnull().astype(int)
    
    # Fill missing values by Position
    num_cols = ['Age', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    for col in num_cols:
        df[col] = df.groupby('Position')[col].transform(lambda x: x.fillna(x.mean()))
        df[col] = df[col].fillna(df[col].mean())
    
    # Physical Attributes
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Height_Weight_Ratio'] = df['Height'] / df['Weight']
    
    # Performance Indices
    df['Power_Index'] = df['Weight'] * df['Vertical_Jump']
    df['Speed_Power'] = df['Weight'] / df['Sprint_40yd']
    df['Explosiveness'] = (df['Vertical_Jump'] + df['Broad_Jump']) / df['Sprint_40yd']
    
    # Position-relative performance
    for col in ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps']:
        df[f'{col}_pos_mean_diff'] = df[col] - df.groupby('Position')[col].transform('mean')
    
    # Year-relative performance
    for col in ['Sprint_40yd', 'Vertical_Jump']:
        df[f'{col}_year_mean_diff'] = df[col] - df.groupby('Year')[col].transform('mean')

    return df

train_df = engineer_features(train)
test_df = engineer_features(test)

X = train_df.drop(['Id', 'Drafted'], axis=1)
y = train_df['Drafted']
X_test = test_df.drop(['Id'], axis=1)

cat_cols = ['School', 'Player_Type', 'Position_Type', 'Position']
for col in cat_cols:
    X[col] = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

# 10-fold CV for stability
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
final_preds = np.zeros(len(test))
cv_scores = []

for i, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=5,
        random_strength=2,
        bagging_temperature=0.2,
        eval_metric='AUC',
        random_seed=42 + i,
        verbose=0
    )
    
    model.fit(X_train, y_train, cat_features=cat_cols, eval_set=(X_val, y_val), early_stopping_rounds=200)
    
    preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, preds)
    cv_scores.append(score)
    
    final_preds += model.predict_proba(X_test)[:, 1] / 10

print(f"Mean CV AUC: {np.mean(cv_scores):.4f}")

# Final submission file
submission = pd.DataFrame({'Id': test['Id'], 'Drafted': final_preds})
submission.to_csv('test_predictions.csv', index=False)
