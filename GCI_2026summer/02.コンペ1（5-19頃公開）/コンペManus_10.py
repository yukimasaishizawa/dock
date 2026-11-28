import pandas as pd
import numpy as np
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

def preprocess(df, is_train=True, meta=None):
    df = df.copy()
    # Physical
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    df['Speed_Score'] = (df['Weight'] * 200) / (df['Sprint_40yd'] ** 4)
    
    num_cols = ['Age', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    
    if is_train:
        meta = {}
        meta['medians'] = df.groupby('Position')[num_cols].median().to_dict()
        meta['global_medians'] = df[num_cols].median().to_dict()
        global_mean = df['Drafted'].mean()
        counts = df.groupby('School')['Drafted'].count()
        means = df.groupby('School')['Drafted'].mean()
        m = 10
        school_te = (counts * means + m * global_mean) / (counts + m)
        meta['school_te'] = school_te.to_dict()
        meta['global_mean'] = global_mean
        le_dict = {}
        for col in ['Player_Type', 'Position_Type', 'Position']:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
        meta['le_dict'] = le_dict
    
    for col in num_cols:
        df[col] = df.apply(lambda row: meta['medians'][col].get(row['Position'], meta['global_medians'][col]) if pd.isnull(row[col]) else row[col], axis=1)
        df[col] = df[col].fillna(meta['global_medians'][col])
    
    df['School_TE'] = df['School'].map(meta['school_te']).fillna(meta['global_mean'])
    
    for col in ['Player_Type', 'Position_Type', 'Position']:
        le = meta['le_dict'][col]
        df[col] = df[col].astype(str).map(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])
        
    return df, meta

if __name__ == "__main__":
    # Load
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    # Step 1: Initial model for Pseudo-labeling
    X_tr_p, meta_p = preprocess(train_df, is_train=True)
    X_te_p, _ = preprocess(test_df, is_train=False, meta=meta_p)
    X_tr_init = X_tr_p.drop(['Id', 'Drafted', 'School'], axis=1)
    y_tr_init = X_tr_p['Drafted']
    X_te_init = X_te_p.drop(['Id', 'School'], axis=1)[X_tr_init.columns]

    m_init = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, random_state=42, verbosity=-1)
    m_init.fit(X_tr_init, y_tr_init)
    test_probs = m_init.predict_proba(X_te_init)[:, 1]

    # Step 2: Combine with Pseudo-labels
    test_df['Drafted'] = test_probs
    pseudo = test_df[(test_df['Drafted'] > 0.95) | (test_df['Drafted'] < 0.05)].copy()
    pseudo['Drafted'] = (pseudo['Drafted'] > 0.5).astype(int)
    combined_train = pd.concat([train_df, pseudo], axis=0).reset_index(drop=True)

    # Step 3: Final Ensemble
    X_comb_p, meta_f = preprocess(combined_train, is_train=True)
    X_te_f, _ = preprocess(test_df.drop('Drafted', axis=1), is_train=False, meta=meta_f)
    X_final = X_comb_p.drop(['Id', 'Drafted', 'School'], axis=1)
    y_final = X_comb_p['Drafted']
    X_test_final = X_te_f.drop(['Id', 'School'], axis=1)[X_final.columns]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    final_preds = np.zeros(len(test_df))

    for tr_idx, val_idx in skf.split(X_final, y_final):
        X_tr, X_val = X_final.iloc[tr_idx], X_final.iloc[val_idx]
        y_tr, y_val = y_final.iloc[tr_idx], y_final.iloc[val_idx]
        
        # Ensemble of 3 models
        m1 = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.02, num_leaves=63, random_state=42, verbosity=-1)
        m1.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(100)])
        
        m2 = CatBoostClassifier(iterations=1000, learning_rate=0.02, depth=7, random_seed=42, verbose=False, early_stopping_rounds=100)
        m2.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        
        m3 = XGBClassifier(n_estimators=1000, learning_rate=0.02, max_depth=7, random_state=42, early_stopping_rounds=100)
        m3.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        
        final_preds += (m1.predict_proba(X_test_final)[:, 1] * 0.4 + 
                        m2.predict_proba(X_test_final)[:, 1] * 0.4 + 
                        m3.predict_proba(X_test_final)[:, 1] * 0.2) / 5

    submission = pd.DataFrame({'Id': test_df['Id'], 'Drafted': final_preds})
    submission.to_csv('submission.csv', index=False)
    print("Final submission.csv created with Pseudo-labeling and Ensemble.")
