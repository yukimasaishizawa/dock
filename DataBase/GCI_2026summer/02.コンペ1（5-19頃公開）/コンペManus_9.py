import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

def preprocess(df, is_train=True, meta=None):
    df = df.copy()
    # Physical
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Speed_Score'] = (df['Weight'] * 200) / (df['Sprint_40yd'] ** 4)
    df['Power_Index'] = df['Weight'] * df['Vertical_Jump'] / (df['Sprint_40yd'] + 1e-6)
    
    # Ranks
    combine_cols = ['Vertical_Jump', 'Broad_Jump', 'Bench_Press_Reps', 'Sprint_40yd']
    for col in combine_cols:
        df[f'{col}_rank'] = df.groupby('Position')[col].rank(pct=True)
    
    if is_train:
        meta = {}
        # School TE
        global_mean = df['Drafted'].mean()
        school_stats = df.groupby('School')['Drafted'].agg(['count', 'mean'])
        m = 10
        school_te = (school_stats['count'] * school_stats['mean'] + m * global_mean) / (school_stats['count'] + m)
        meta['school_te'] = school_te.to_dict()
        meta['global_mean'] = global_mean
        
        # Medians
        medians = {}
        for col in ['Age', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']:
            pos_med = df.groupby('Position')[col].median().to_dict()
            overall_med = df[col].median()
            df[col] = df.apply(lambda row: pos_med.get(row['Position'], overall_med) if pd.isnull(row[col]) else row[col], axis=1)
            medians[col] = (pos_med, overall_med)
        meta['medians'] = medians
        
        # Encoding
        le_dict = {}
        for col in ['Player_Type', 'Position_Type', 'Position']:
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
        return df, meta

if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    X_full, meta = preprocess(train_df, is_train=True)
    X = X_full.drop(['Id', 'Drafted', 'School'], axis=1)
    y = X_full['Drafted']

    X_test_proc, _ = preprocess(test_df, is_train=False, meta=meta)
    X_test = X_test_proc.drop(['Id', 'School'], axis=1)[X.columns]

    # Best Params
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'verbosity': -1,
        'learning_rate': 0.045,
        'num_leaves': 75,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 50,
        'random_state': 42
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_preds = np.zeros(len(X_test))

    for tr_idx, val_idx in skf.split(X, y):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
        
        m_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=1000)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100)])
        
        m_cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, random_seed=42, verbose=False, early_stopping_rounds=100)
        m_cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        
        test_preds += (m_lgb.predict_proba(X_test)[:, 1] * 0.5 + m_cat.predict_proba(X_test)[:, 1] * 0.5) / 5

    submission = pd.DataFrame({'Id': test_df['Id'], 'Drafted': test_preds})
    submission.to_csv('submission.csv', index=False)
    print("Final submission.csv created.")
