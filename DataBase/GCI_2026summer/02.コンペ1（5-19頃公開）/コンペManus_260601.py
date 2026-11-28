import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TRAIN_PATH = '/home/ubuntu/upload/train.csv'
TEST_PATH = '/home/ubuntu/upload/test.csv'

class AdvancedFeatureMaker:
    def __init__(self, smoothing=30):
        self.smoothing = smoothing
        self.global_rate_ = None
        self.target_maps_ = {}
        self.group_stats_ = {}
        self.numeric_medians_ = {}
        self.feature_columns_ = None
        self.cat_cols = ['School', 'Player_Type', 'Position_Type', 'Position']

    def _base_engineer(self, df):
        out = df.copy()
        for c in self.cat_cols:
            out[c] = out[c].astype(str).fillna('Unknown').str.strip()
        
        # 必須計測値
        h = out['Height'].replace(0, np.nan)
        w = out['Weight'].replace(0, np.nan)
        sprint = out['Sprint_40yd'].replace(0, np.nan)
        vj = out['Vertical_Jump'].replace(0, np.nan)
        bj = out['Broad_Jump'].replace(0, np.nan)
        bp = out['Bench_Press_Reps'].replace(0, np.nan)
        
        # 指標作成
        out['BMI'] = w / (h ** 2)
        out['Weight_lb'] = w * 2.20462
        out['Speed_Score'] = (out['Weight_lb'] * 200) / (sprint ** 4)
        out['Explosiveness'] = (vj / 2.54) + (bj / 2.54 / 12.0)
        out['Size_Speed'] = out['BMI'] / sprint
        out['Power_Factor'] = bp * out['Weight_lb'] / 100.0
        
        # 欠損フラグ（NFLでは欠損自体が重要な意味を持つことが多い）
        for c in ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle', 'Age']:
            out[f'is_missing_{c}'] = out[c].isna().astype(int)
        
        # カテゴリ交互作用
        out['Pos_Type_Full'] = out['Position_Type'] + "_" + out['Position']
        # Drafted列が残っている場合は削除（予測時のエラー防止）
        if 'Drafted' in out.columns:
            out = out.drop(columns=['Drafted'])
        return out

    def fit(self, df, y):
        base = self._base_engineer(df)
        self.global_rate_ = float(np.mean(y))
        
        # Target Encoding (with smoothing)
        te_cols = self.cat_cols + ['Pos_Type_Full']
        for c in te_cols:
            agg = pd.DataFrame({'cat': base[c], 'y': y}).groupby('cat')['y'].agg(['sum', 'count'])
            self.target_maps_[c] = ((agg['sum'] + self.smoothing * self.global_rate_) / (agg['count'] + self.smoothing)).to_dict()
        
        # Position-based Z-scores (汎化性能に極めて重要)
        num_cols = ['Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle', 'BMI', 'Speed_Score', 'Explosiveness']
        for group_col in ['Position', 'Position_Type']:
            self.group_stats_[group_col] = {}
            for n in num_cols:
                stats = base.groupby(group_col)[n].agg(['mean', 'std'])
                self.group_stats_[group_col][n] = stats.to_dict('index')
        
        self.numeric_medians_ = {n: base[n].median() for n in num_cols if n in base.columns}
        self.feature_columns_ = self.transform(df).columns.tolist()
        return self

    def transform(self, df):
        out = self._base_engineer(df)
        
        # Apply TE
        te_cols = self.cat_cols + ['Pos_Type_Full']
        for c in te_cols:
            out[f'{c}_te'] = out[c].map(self.target_maps_.get(c, {})).fillna(self.global_rate_)
            
        # Apply Z-scores
        for group_col, stats_map in self.group_stats_.items():
            for n, stats in stats_map.items():
                if n in out.columns:
                    m = out[group_col].map(lambda x: stats.get(x, {}).get('mean', self.numeric_medians_.get(n, 0)))
                    s = out[group_col].map(lambda x: stats.get(x, {}).get('std', 1.0)).replace(0, 1.0).fillna(1.0)
                    out[f'{n}_z_{group_col}'] = (out[n] - m) / s

        # Drop raw categorical strings for models that don't handle them
        cols_to_drop = self.cat_cols + ['Pos_Type_Full', 'Id', 'Year']
        return out.drop(columns=[c for c in cols_to_drop if c in out.columns], errors='ignore')

def train_eval(X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    oof_preds = {
        'cat': np.zeros(len(X)),
        'xgb': np.zeros(len(X)),
        'lgb': np.zeros(len(X)),
        'et': np.zeros(len(X))
    }
    
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
        
        # CatBoost
        cat = CatBoostClassifier(iterations=1000, depth=3, learning_rate=0.01, l2_leaf_reg=20, 
                                 eval_metric='AUC', random_seed=RANDOM_STATE, verbose=False)
        cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
        oof_preds['cat'][va_idx] = cat.predict_proba(X_va)[:, 1]
        
        # XGBoost
        model_xgb = xgb.XGBClassifier(n_estimators=1000, max_depth=3, learning_rate=0.01, 
                                      subsample=0.7, colsample_bytree=0.7, 
                                      eval_metric='auc', random_state=RANDOM_STATE,
                                      early_stopping_rounds=100)
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_preds['xgb'][va_idx] = model_xgb.predict_proba(X_va)[:, 1]
        
        # LightGBM
        model_lgb = lgb.LGBMClassifier(n_estimators=1000, max_depth=4, learning_rate=0.01,
                                       num_leaves=15, subsample=0.8, colsample_bytree=0.8, 
                                       importance_type='gain', random_state=RANDOM_STATE, verbose=-1)
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        oof_preds['lgb'][va_idx] = model_lgb.predict_proba(X_va)[:, 1]
        
        # ExtraTrees
        et = ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5, 
                                  random_state=RANDOM_STATE, n_jobs=-1)
        et.fit(X_tr.fillna(-999), y_tr)
        oof_preds['et'][va_idx] = et.predict_proba(X_va.fillna(-999))[:, 1]
        
        print(f"Fold {fold+1} completed.")

    # Find best weights for ensemble
    best_auc = 0
    best_w = (0, 0, 0, 0)
    for w_cat in np.linspace(0, 0.6, 7):
        for w_xgb in np.linspace(0, 0.6, 7):
            for w_lgb in np.linspace(0, 0.6, 7):
                w_et = 1.0 - (w_cat + w_xgb + w_lgb)
                if w_et < -0.001 or w_et > 1.001: continue
                p = w_cat*oof_preds['cat'] + w_xgb*oof_preds['xgb'] + w_lgb*oof_preds['lgb'] + w_et*oof_preds['et']
                auc = roc_auc_score(y, p)
                if auc > best_auc:
                    best_auc = auc
                    best_w = (w_cat, w_xgb, w_lgb, w_et)
    
    print(f"\nOOF AUC Scores:")
    for k, v in oof_preds.items():
        print(f"{k}: {roc_auc_score(y, v):.5f}")
    print(f"Ensemble (best weights {best_w}): {best_auc:.5f}")
    return best_auc, best_w

def main():
    df = pd.read_csv(TRAIN_PATH)
    y = df['Drafted']
    
    print("Starting re-training for OOF AUC 0.85+...")
    fm = AdvancedFeatureMaker().fit(df, y)
    X = fm.transform(df)
    
    auc, weights = train_eval(X, y)
    
    # Final training on all data
    print("\nTraining final models on full data...")
    cat = CatBoostClassifier(iterations=1000, depth=4, learning_rate=0.02, l2_leaf_reg=10, random_seed=RANDOM_STATE, verbose=False).fit(X, y)
    model_xgb = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.7, colsample_bytree=0.7, random_state=RANDOM_STATE).fit(X, y)
    model_lgb = lgb.LGBMClassifier(n_estimators=500, max_depth=3, learning_rate=0.01, subsample=0.7, colsample_bytree=0.7, random_state=RANDOM_STATE, verbose=-1).fit(X, y)
    et = ExtraTreesClassifier(n_estimators=500, max_depth=8, min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=-1).fit(X.fillna(-999), y)
    
    if os.path.exists(TEST_PATH):
        test_df = pd.read_csv(TEST_PATH)
        X_test = fm.transform(test_df)
        p_test = (weights[0]*cat.predict_proba(X_test)[:, 1] + 
                  weights[1]*model_xgb.predict_proba(X_test)[:, 1] + 
                  weights[2]*model_lgb.predict_proba(X_test)[:, 1] + 
                  weights[3]*et.predict_proba(X_test.fillna(-999))[:, 1])
        
        sub = pd.DataFrame({'Id': test_df['Id'], 'Drafted': p_test})
        sub.to_csv('/home/ubuntu/drafted_model/test_submission_v2.csv', index=False)
        print(f"Final submission saved with OOF AUC: {auc:.5f}")

if __name__ == "__main__":
    main()
