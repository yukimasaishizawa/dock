import os
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TRAIN_PATH = '/home/ubuntu/upload/train.csv'
TEST_PATH = '/home/ubuntu/upload/test.csv'

class RobustFeatureMaker:
    def __init__(self, smoothing=50):
        self.smoothing = smoothing
        self.global_rate_ = None
        self.target_maps_ = {}
        self.group_stats_ = {}
        self.numeric_medians_ = {}
        self.cat_cols = ['Player_Type', 'Position_Type', 'Position'] # Schoolは過学習の元なので除外

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
        
        # 頑健な指標のみ採用
        out['BMI'] = w / (h ** 2)
        out['Weight_lb'] = w * 2.20462
        out['Speed_Score'] = (out['Weight_lb'] * 200) / (sprint ** 4)
        out['Explosiveness'] = (vj / 2.54) + (bj / 2.54 / 12.0)
        
        # 交互作用（NFLで重視される指標）
        out['Weight_Sprint'] = out['Weight_lb'] / sprint
        out['Height_Weight'] = h * w
        out['Power_Explosion'] = bp * out['Explosiveness']
        
        # 欠損パターンは非常に重要
        for c in ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']:
            out[f'is_missing_{c}'] = out[c].isna().astype(int)
        
        # カテゴリ交互作用（ポジションと体格の関係は重要）
        out['Pos_Type_Full'] = out['Position_Type'] + "_" + out['Position']
        # 文字列列は削除（TEなどで数値化済みのため）
        out = out.drop(columns=['Pos_Type_Full'])
        return out

    def fit(self, df, y):
        # TE用に元のPos_Type_Fullが必要なので一時的に保持
        base_for_te = df.copy()
        for c in self.cat_cols: base_for_te[c] = base_for_te[c].astype(str).fillna('Unknown')
        base_for_te['Pos_Type_Full'] = base_for_te['Position_Type'] + "_" + base_for_te['Position']
        
        base = self._base_engineer(df)
        self.global_rate_ = float(np.mean(y))
        
        # Target Encoding (より強いスムージング)
        te_target_cols = self.cat_cols + ['Pos_Type_Full']
        for c in te_target_cols:
            agg = pd.DataFrame({'cat': base_for_te[c], 'y': y}).groupby('cat')['y'].agg(['sum', 'count'])
            self.target_maps_[c] = ((agg['sum'] + self.smoothing * self.global_rate_) / (agg['count'] + self.smoothing)).to_dict()
        
        # Position-based Z-scores (必須)
        num_cols = ['Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'BMI', 'Speed_Score', 'Explosiveness']
        for group_col in ['Position', 'Position_Type']:
            self.group_stats_[group_col] = {}
            for n in num_cols:
                stats = base.groupby(group_col)[n].agg(['mean', 'std'])
                self.group_stats_[group_col][n] = stats.to_dict('index')
        
        self.numeric_medians_ = {n: base[n].median() for n in num_cols if n in base.columns}
        return self

    def transform(self, df):
        # TE用に元のPos_Type_Fullを一時的に作成
        df_for_te = df.copy()
        for c in self.cat_cols: df_for_te[c] = df_for_te[c].astype(str).fillna('Unknown')
        df_for_te['Pos_Type_Full'] = df_for_te['Position_Type'] + "_" + df_for_te['Position']
        
        out = self._base_engineer(df)
        te_target_cols = self.cat_cols + ['Pos_Type_Full']
        for c in te_target_cols:
            out[f'{c}_te'] = df_for_te[c].map(self.target_maps_.get(c, {})).fillna(self.global_rate_)
        for group_col, stats_map in self.group_stats_.items():
            for n, stats in stats_map.items():
                if n in out.columns:
                    m = out[group_col].map(lambda x: stats.get(x, {}).get('mean', self.numeric_medians_.get(n, 0)))
                    s = out[group_col].map(lambda x: stats.get(x, {}).get('std', 1.0)).replace(0, 1.0).fillna(1.0)
                    out[f'{n}_z_{group_col}'] = (out[n] - m) / s
        
        # 特徴量の順序と有無を固定
        # 学校は頻度のみ採用し、名前自体は除外（過学習防止）
        out['School_freq'] = out['School'].map(out['School'].value_counts() / len(out)).fillna(0)
        
        # 年齢と年度は数値として維持
        cols_to_drop = self.cat_cols + ['Id', 'School', 'Drafted']
        final_df = out.drop(columns=[c for c in cols_to_drop if c in out.columns], errors='ignore')
        if hasattr(self, 'final_columns'):
            for c in self.final_columns:
                if c not in final_df.columns: final_df[c] = np.nan
            return final_df[self.final_columns]
        self.final_columns = final_df.columns.tolist()
        return final_df

def get_models():
    return {
        'lgb': lgb.LGBMClassifier(n_estimators=2000, max_depth=5, learning_rate=0.005, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5, random_state=RANDOM_STATE, verbose=-1),
        'xgb': xgb.XGBClassifier(n_estimators=2000, max_depth=5, learning_rate=0.005, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5, reg_lambda=0.5, random_state=RANDOM_STATE, early_stopping_rounds=200),
        'cat': CatBoostClassifier(iterations=2000, depth=5, learning_rate=0.005, l2_leaf_reg=20, random_seed=RANDOM_STATE, verbose=False, early_stopping_rounds=200),
        'et': ExtraTreesClassifier(n_estimators=1000, max_depth=12, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
        'rf': RandomForestClassifier(n_estimators=1000, max_depth=12, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1)
    }

def train_with_pseudo(X_train, y_train, X_test, model_name, model):
    # 1. 学習データで学習
    if model_name == 'xgb':
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros(len(X_train))
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = get_models()['xgb']
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], eval_set=[(X_train.iloc[va_idx], y_train.iloc[va_idx])], verbose=False)
            oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        model.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    elif model_name == 'lgb':
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros(len(X_train))
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = get_models()['lgb']
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], eval_set=[(X_train.iloc[va_idx], y_train.iloc[va_idx])], callbacks=[lgb.early_stopping(100)])
            oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        model.fit(X_train, y_train)
    elif model_name == 'cat':
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros(len(X_train))
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = get_models()['cat']
            m.fit(X_train.iloc[tr_idx], y_train.iloc[tr_idx], eval_set=(X_train.iloc[va_idx], y_train.iloc[va_idx]))
            oof[va_idx] = m.predict_proba(X_train.iloc[va_idx])[:, 1]
        model.fit(X_train, y_train)
    elif model_name == 'rf':
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros(len(X_train))
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = get_models()['rf']
            m.fit(X_train.iloc[tr_idx].fillna(-999), y_train.iloc[tr_idx])
            oof[va_idx] = m.predict_proba(X_train.iloc[va_idx].fillna(-999))[:, 1]
        model.fit(X_train.fillna(-999), y_train)
    else:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        oof = np.zeros(len(X_train))
        for tr_idx, va_idx in skf.split(X_train, y_train):
            m = get_models()['et']
            m.fit(X_train.iloc[tr_idx].fillna(-999), y_train.iloc[tr_idx])
            oof[va_idx] = m.predict_proba(X_train.iloc[va_idx].fillna(-999))[:, 1]
        model.fit(X_train.fillna(-999), y_train)
    
    # 2. テストデータへの予測
    if model_name == 'et':
        test_preds = model.predict_proba(X_test.fillna(-999))[:, 1]
    else:
        test_preds = model.predict_proba(X_test)[:, 1]
    
    # 3. 擬似ラベルの作成 (確信度の高いもののみ)
    high_conf_idx = (test_preds > 0.9) | (test_preds < 0.1)
    X_pseudo = X_test[high_conf_idx]
    y_pseudo = (test_preds[high_conf_idx] > 0.5).astype(int)
    
    # 4. 拡張データで再学習
    X_ext = pd.concat([X_train, X_pseudo])
    y_ext = pd.concat([y_train, pd.Series(y_pseudo)])
    
    if model_name == 'et':
        model.fit(X_ext.fillna(-999), y_ext)
    elif model_name == 'xgb':
        # 早期終了を無効化して再学習
        model.set_params(early_stopping_rounds=None)
        model.fit(X_ext, y_ext, verbose=False)
    else:
        model.fit(X_ext, y_ext)
    
    return oof, model

def main():
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    y = train['Drafted']
    
    fm = RobustFeatureMaker().fit(train, y)
    X = fm.transform(train)
    X_test = fm.transform(test)
    
    print(f"Feature engineering done. Features: {X.shape[1]}")
    
    models = get_models()
    oof_df = pd.DataFrame()
    final_test_preds = pd.DataFrame()
    
    for name, model in models.items():
        print(f"Training {name} with Pseudo-Labeling...")
        oof, trained_model = train_with_pseudo(X, y, X_test, name, model)
        oof_df[name] = oof
        if name in ['et', 'rf']:
            final_test_preds[name] = trained_model.predict_proba(X_test.fillna(-999))[:, 1]
        else:
            final_test_preds[name] = trained_model.predict_proba(X_test)[:, 1]
        print(f"{name} OOF AUC: {roc_auc_score(y, oof):.5f}")

    # Stacking (Ridge Regression for better calibration)
    from sklearn.linear_model import Ridge
    print("\nStacking with Ridge Regression...")
    stacker = Ridge(alpha=1.0)
    stacker.fit(oof_df, y)
    
    test_probs = stacker.predict(final_test_preds)
    # 確率を0-1にクリップ
    test_probs = np.clip(test_probs, 0.0001, 0.9999)
    
    sub = pd.DataFrame({'Id': test['Id'], 'Drafted': test_probs})
    sub.to_csv('/home/ubuntu/drafted_model/test_submission_v3.csv', index=False)
    
    print(f"Final OOF Stacking AUC: {roc_auc_score(y, stacker.predict(oof_df)):.5f}")
    print("V3 submission saved.")

if __name__ == "__main__":
    main()
