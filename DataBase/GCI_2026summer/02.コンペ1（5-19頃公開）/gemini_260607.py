import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
import warnings
import os

warnings.filterwarnings('ignore')

def build_features(df_train, df_test):
    """
    データの特徴量をエンジニアリングする関数。
    TrainとTestのデータ間のリークを防ぐため、カテゴリ別集計などは頑健に行います。
    """
    # 処理のために一時的に結合
    df_train['is_train'] = 1
    df_test['is_train'] = 0
    df_test['Drafted'] = np.nan
    
    combined = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    
    # 1. 身体サイズに関する特徴量
    # BMIに類似した体格バランス指数 (体重 kg / 身長 m^2)
    # ※元のデータでHeightはメートル(1.8mなど)、Weightはキログラム(90kgなど)で入っていると仮定
    combined['BMI'] = combined['Weight'] / (combined['Height'] ** 2 + 1e-8)
    combined['Weight_to_Height'] = combined['Weight'] / (combined['Height'] + 1e-8)
    
    # 2. 各パフォーマンステストの欠損有無フラグ (ドラフト選考において測定不参加は重要なシグナルとなるため)
    performance_cols = ['Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
    for col in performance_cols:
        combined[f'{col}_isna'] = combined[col].isna().astype(int)
        
    # 3. ポジション別 (Position_Type) のパフォーマンステスト標準化 (偏差値化)
    # NFLではポジションごとに求められる身体能力基準が劇的に異なるため、全体平均ではなくポジション内相対評価にする
    for col in performance_cols + ['Weight', 'Height', 'BMI']:
        # グループごとの平均と標準偏差を計算
        group_stats = combined.groupby('Position_Type')[col].transform('mean')
        group_std = combined.groupby('Position_Type')[col].transform('std')
        
        # ポジション内での偏差 (Z-score)
        combined[f'{col}_pos_z'] = (combined[col] - group_stats) / (group_std + 1e-8)
        # 欠損値は0 (平均値) で補完
        combined[f'{col}_pos_z'] = combined[f'{col}_pos_z'].fillna(0)
    
    # 4. カテゴリカル変数の処理 (School, Player_Type, Position_Type, Position)
    categorical_cols = ['School', 'Player_Type', 'Position_Type', 'Position']
    
    # 各カテゴリの出現頻度 (頻出大学などの強さを表す)
    for col in categorical_cols:
        freq = combined[col].value_counts().to_dict()
        combined[f'{col}_freq'] = combined[col].map(freq)
        
    # ラベルエンコーディング
    for col in categorical_cols:
        le = LabelEncoder()
        # 欠損値を文字列に置換してエンコード
        combined[col] = combined[col].fillna('Missing')
        combined[col] = le.fit_transform(combined[col].astype(str))
        
    # 5. 年代特徴量
    combined['Year_diff'] = combined['Year'] - 2000  # 年代トレンドの反映
    
    # 元の形に分離
    train_features = combined[combined['is_train'] == 1].drop(columns=['is_train'])
    test_features = combined[combined['is_train'] == 0].drop(columns=['is_train', 'Drafted'])
    
    return train_features, test_features

def main():
    print("=== [1] データのロード ===")
    # ローカルのアップロードされたファイルを読み込みます
    train_path = 'train.csv'
    test_path = 'test.csv'
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("train.csv または test.csv が見つかりません。パスを確認してください。")
        
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # テストデータのID列を保持
    test_ids = test['Id']
    
    print(f"訓練データ数: {train.shape[0]}, 特徴量数: {train.shape[1]}")
    print(f"テストデータ数: {test.shape[0]}, 特徴量数: {test.shape[1]}")
    
    print("\n=== [2] 特徴量エンジニアリングの実行 ===")
    train_feat, test_feat = build_features(train, test)
    
    # 不要なカラムの削除
    features_to_drop = ['Id', 'Drafted']
    X = train_feat.drop(columns=features_to_drop)
    y = train_feat['Drafted'].astype(int)
    X_test = test_feat.drop(columns=['Id'])
    
    print(f"モデル学習に使用する最終特徴量数: {X.shape[1]}")
    print(f"特徴量リスト: {list(X.columns)}")
    
    # 5-Fold Stratified K-Fold のセットアップ
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 各モデルの予測値格納用
    oof_lgb = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    
    preds_lgb = np.zeros(len(X_test))
    preds_xgb = np.zeros(len(X_test))
    preds_cat = np.zeros(len(X_test))
    
    print("\n=== [3] モデルの学習 & クロスバリデーション ===")
    
    # カテゴリカル特徴量のインデックス (CatBoost用)
    cat_features = ['School', 'Player_Type', 'Position_Type', 'Position']
    cat_indices = [X.columns.get_loc(col) for col in cat_features]
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold + 1} / {n_splits} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # -----------------------------------------------------
        # 1. LightGBM モデル
        # -----------------------------------------------------
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'learning_rate': 0.03,
            'num_leaves': 31,
            'max_depth': 6,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'min_child_samples': 20,
            'seed': 42 + fold,
            'verbose': -1
        }
        
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        lgb_model = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=1500,
            valid_sets=[train_data, val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        
        oof_lgb[val_idx] = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
        preds_lgb += lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration) / n_splits
        
        # -----------------------------------------------------
        # 2. XGBoost モデル
        # -----------------------------------------------------
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'learning_rate': 0.03,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'seed': 42 + fold,
            'tree_method': 'hist'
        }
        
        xgb_train = xgb.DMatrix(X_train, label=y_train)
        xgb_val = xgb.DMatrix(X_val, label=y_val)
        xgb_test = xgb.DMatrix(X_test)
        
        xgb_model = xgb.train(
            xgb_params,
            xgb_train,
            num_boost_round=1500,
            evals=[(xgb_train, 'train'), (xgb_val, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        oof_xgb[val_idx] = xgb_model.predict(xgb_val, iteration_range=(0, xgb_model.best_iteration + 1))
        preds_xgb += xgb_model.predict(xgb_test, iteration_range=(0, xgb_model.best_iteration + 1)) / n_splits
        
        # -----------------------------------------------------
        # 3. CatBoost モデル
        # -----------------------------------------------------
        cat_model = CatBoostClassifier(
            iterations=1500,
            learning_rate=0.03,
            depth=6,
            eval_metric='AUC',
            random_seed=42 + fold,
            verbose=False,
            early_stopping_rounds=50
        )
        
        # CatBoost用のPoolを作成 (カテゴリカル変数を明示的に指定)
        train_pool = Pool(X_train, y_train, cat_features=cat_indices)
        val_pool = Pool(X_val, y_val, cat_features=cat_indices)
        test_pool = Pool(X_test, cat_features=cat_indices)
        
        cat_model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        
        oof_cat[val_idx] = cat_model.predict_proba(val_pool)[:, 1]
        preds_cat += cat_model.predict_proba(test_pool)[:, 1] / n_splits
        
        # 個別フォールドでのスコア確認
        fold_lgb_auc = roc_auc_score(y_val, oof_lgb[val_idx])
        fold_xgb_auc = roc_auc_score(y_val, oof_xgb[val_idx])
        fold_cat_auc = roc_auc_score(y_val, oof_cat[val_idx])
        fold_ensemble = roc_auc_score(y_val, (oof_lgb[val_idx] + oof_xgb[val_idx] + oof_cat[val_idx]) / 3)
        
        print(f"Fold {fold+1} Scores -> LGB: {fold_lgb_auc:.5f} | XGB: {fold_xgb_auc:.5f} | Cat: {fold_cat_auc:.5f} | 3-Model Ensemble: {fold_ensemble:.5f}")

    # 全体でのOOFスコア
    auc_lgb = roc_auc_score(y, oof_lgb)
    auc_xgb = roc_auc_score(y, oof_xgb)
    auc_cat = roc_auc_score(y, oof_cat)
    
    # 簡易平均アンサンブル
    oof_ensemble = (oof_lgb + oof_xgb + oof_cat) / 3.0
    auc_ensemble = roc_auc_score(y, oof_ensemble)
    
    print("\n" + "="*40)
    print("=== [4] クロスバリデーション(OOF) 最終集計スコア ===")
    print(f"  LightGBM 単体 AUC : {auc_lgb:.5f}")
    print(f"  XGBoost  単体 AUC : {auc_xgb:.5f}")
    print(f"  CatBoost 単体 AUC : {auc_cat:.5f}")
    print(f"  3モデル平均アンサンブル AUC : {auc_ensemble:.5f}")
    print("="*40)
    
    # テストデータの予測値（3モデルのブレンド）
    test_preds = (preds_lgb + preds_xgb + preds_cat) / 3.0
    
    # 提出ファイルの作成
    submission = pd.DataFrame({
        'Id': test_ids,
        'Drafted': test_preds
    })
    
    submission_file = 'submission.csv'
    submission.to_csv(submission_file, index=False)
    print(f"\n予測が正常に完了しました！結果を '{submission_file}' に保存しました。")
    print(submission.head())

if __name__ == '__main__':
    main()
```
eof

### コードの使い方
1. お手元のPython環境に、以下のパッケージをインストールします（未インストールの場合）。
   ```bash
   pip install pandas numpy scikit-learn lightgbm xgboost catboost
   ```
2. このプログラムと同じフォルダに、アップロードしていただいた `train.csv` と `test.csv` を配置します。
3. 作成されたスクリプトを実行します。
   ```bash
   python nfl_draft_predictor.py