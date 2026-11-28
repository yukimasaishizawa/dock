import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def preprocess(df, is_train=True, meta=None):
    df = df.copy()
    
    # 1. Physical interactions
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Speed_Score'] = (df['Weight'] * 200) / (df['Sprint_40yd'] ** 4)
    df['Power_Index'] = df['Weight'] * df['Vertical_Jump'] / (df['Sprint_40yd'] + 1e-6)
    
    # 2. Position Groups
    pos_map = {
        'QB': 'Backs', 'RB': 'Backs', 'WR': 'Receivers', 'TE': 'Receivers',
        'OT': 'OL', 'OG': 'OL', 'C': 'OL',
        'DE': 'DL', 'DT': 'DL', 'NT': 'DL',
        'ILB': 'LB', 'OLB': 'LB',
        'CB': 'DB', 'S': 'DB', 'FS': 'DB', 'SS': 'DB',
        'K': 'ST', 'P': 'ST', 'LS': 'ST'
    }
    df['Pos_Group'] = df['Position'].map(pos_map).fillna('Other')
    
    # 3. Combine Rank based athleticism
    combine_cols = ['Vertical_Jump', 'Broad_Jump', 'Bench_Press_Reps', 'Sprint_40yd']
    for col in combine_cols:
        df[f'{col}_rank'] = df.groupby('Position')[col].rank(pct=True)
    df['Athleticism'] = df[[f'{c}_rank' for c in combine_cols]].mean(axis=1)

    # 4. Missing flags
    for col in ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']:
        df[f'{col}_nan'] = df[col].isnull().astype(int)

    if is_train:
        meta = {}
        # School Target Encoding
        global_mean = df['Drafted'].mean()
        school_counts = df.groupby('School')['Drafted'].count()
        school_means = df.groupby('School')['Drafted'].mean()
        m = 10
        school_te = (school_counts * school_means + m * global_mean) / (school_counts + m)
        df['School_TE'] = df['School'].map(school_te)
        meta['school_te'] = school_te.to_dict()
        meta['global_mean'] = global_mean
        
        # Medians for imputation
        medians = {}
        for col in ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']:
            pos_med = df.groupby('Position')[col].median().to_dict()
            overall_med = df[col].median()
            df[col] = df.apply(lambda row: pos_med.get(row['Position'], overall_med) if pd.isnull(row[col]) else row[col], axis=1)
            medians[col] = (pos_med, overall_med)
        meta['medians'] = medians
        
        # Label Encoding
        le_dict = {}
        for col in ['Player_Type', 'Position_Type', 'Position', 'Pos_Group']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
        meta['le_dict'] = le_dict
        
        return df, meta
    else:
        df['School_TE'] = df['School'].map(meta['school_te']).fillna(meta['global_mean'])
        
        medians = meta['medians']
        for col in medians:
            pos_med, overall_med = medians[col]
            df[col] = df.apply(lambda row: pos_med.get(row['Position'], overall_med) if pd.isnull(row[col]) else row[col], axis=1)
            df[col] = df[col].fillna(overall_med)
            
        le_dict = meta['le_dict']
        for col in le_dict:
            le = le_dict[col]
            df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
            df[col] = le.transform(df[col])
            
        return df

if __name__ == "__main__":
    # Load data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # Preprocess
    X_train_full, meta = preprocess(train_df, is_train=True)
    X_test = preprocess(test_df, is_train=False, meta=meta)

    X = X_train_full.drop(['Id', 'Drafted', 'School'], axis=1)
    y = X_train_full['Drafted']
    X_test_final = X_test.drop(['Id', 'School'], axis=1)

    # Best parameters found by Optuna
    best_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'random_state': 42,
        'learning_rate': 0.041410503852920486,
        'num_leaves': 75,
        'feature_fraction': 0.816058197701508,
        'bagging_fraction': 0.48145848810307723,
        'bagging_freq': 2,
        'min_child_samples': 86,
        'lambda_l1': 0.14885592260078295,
        'lambda_l2': 0.37101751774682795
    }

    # Final Train and Predict with 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test_final))

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**best_params, n_estimators=2000)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(200)])
        test_preds += model.predict_proba(X_test_final)[:, 1] / 5

    # Save submission
    submission = pd.DataFrame({'Id': test_df['Id'], 'Drafted': test_preds})
    submission.to_csv('submission.csv', index=False)
    print("Final predictions saved to submission.csv")
