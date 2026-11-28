import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# 回帰(Regressor)から 分類(Classifier) に変更
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, accuracy_score

# ==========================================================
# 【設定：ファイルのパス】
# ==========================================================
TRAIN_DATA_PATH = '/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/train.csv'  # 学習用ファイル名
TEST_DATA_PATH = '/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/test.csv'    # アップロードされた予測用ファイル名

# ==========================================================
# 1. データの読み込み
# ==========================================================
print("データを読み込み中...")
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

test_ids = test_df['Id']
y_train_full = train_df['Drafted'].astype(int) # 確実に0と1の整数にする

X_train_full = train_df.drop(columns=['Id', 'Drafted', 'School'], errors='ignore')
X_test_full = test_df.drop(columns=['Id', 'Drafted', 'School'], errors='ignore')

# 結合して一括前処理
combined = pd.concat([X_train_full, X_test_full], axis=0, ignore_index=True)

# ==========================================================
# 2. 前処理（欠損値補完：平均値＆最頻値 ＆ カテゴリ変換）
# ==========================================================
print("前処理を実行中...")
num_cols = combined.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = combined.select_dtypes(include=['object']).columns.tolist()

# 数値列の補完：平均値
num_imputer = SimpleImputer(strategy='mean')
combined_num = pd.DataFrame(num_imputer.fit_transform(combined[num_cols]), columns=num_cols)

# カテゴリ列の補完：最頻値
cat_imputer = SimpleImputer(strategy='most_frequent')
combined_cat = pd.DataFrame(cat_imputer.fit_transform(combined[cat_cols]), columns=cat_cols)

# 結合とエンコーディング
combined_imputed = pd.concat([combined_num, combined_cat], axis=1)
combined_encoded = pd.get_dummies(combined_imputed, columns=cat_cols, drop_first=True)

X_train_encoded = combined_encoded.iloc[:len(train_df)].copy()
X_test_encoded = combined_encoded.iloc[len(train_df):].copy()

# ==========================================================
# 3. モデルの内部評価（分類の評価指標を使用）
# ==========================================================
X_tr, X_val, y_tr, y_val = train_test_split(X_train_encoded, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

# --- ランダムフォレスト（分類） ---
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_tr, y_tr)
# 「1（指名される）」になる確率を取得
rf_probs = rf_model.predict_proba(X_val)[:, 1]
rf_auc = roc_auc_score(y_val, rf_probs)
print(f"■ ランダムフォレスト AUCスコア: {rf_auc:.4f} (0.5〜1.0の間になり、1に近いほど優秀)")

# --- 勾配ブースティング（分類） ---
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
gb_model.fit(X_tr, y_tr)
# 「1（指名される）」になる確率を取得
gb_probs = gb_model.predict_proba(X_val)[:, 1]
gb_auc = roc_auc_score(y_val, gb_probs)
print(f"■ 勾配ブースティング AUCスコア: {gb_auc:.4f} (0.5〜1.0の間になり、1に近いほど優秀)")

# ==========================================================
# 4. 本番予測（確率として出力するため、負の値は出ません）
# ==========================================================
if gb_auc > rf_auc:
    print("\n→ 勾配ブースティングを採用して本番予測を行います。")
    final_model = gb_model
else:
    print("\n→ ランダムフォレストを採用して本番予測を行います。")
    final_model = rf_model

print("test.csv の Drafted（指名確率）を予測中...")
# 本番データに対して「1になる確率」を予測
test_predictions_proba = final_model.predict_proba(X_test_encoded)[:, 1]

# 結果をデータフレームにまとめる
submission = pd.DataFrame({
    'Id': test_ids,
    'Drafted': test_predictions_proba  # 0.0 〜 1.0 の確率
})

# CSVファイルとして出力
output_filename = 'predicted_result_classification.csv'
submission.to_csv(output_filename, index=False)
print(f"完了しました！ 予測結果は '{output_filename}' に保存されました。")














import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier # 勾配ブースティングのみ使用
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

# ==========================================================
# 【設定：ファイルのパス】※元のパスをそのまま引き継いでいます
# ==========================================================
TRAIN_DATA_PATH = '/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/train.csv'
TEST_DATA_PATH = '/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/test.csv'

# ==========================================================
# 1. データの読み込み
# ==========================================================
print("データを読み込み中...")
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

test_ids = test_df['Id']
y_train_full = train_df['Drafted'].astype(int) # 確実に0と1の整数にする

# 予測に不要・ノイズになりやすい列をドロップ
X_train_full = train_df.drop(columns=['Id', 'Drafted', 'School'], errors='ignore')
X_test_full = test_df.drop(columns=['Id', 'Drafted', 'School'], errors='ignore')

# 結合して一括前処理（列のズレを防ぐため）
combined = pd.concat([X_train_full, X_test_full], axis=0, ignore_index=True)

# ==========================================================
# 2. 前処理（欠損値補完 ＆ カテゴリ変換）
# ==========================================================
print("前処理を実行中...")
num_cols = combined.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols = combined.select_dtypes(include=['object']).columns.tolist()

# 数値列の補完：平均値
num_imputer = SimpleImputer(strategy='mean')
combined_num = pd.DataFrame(num_imputer.fit_transform(combined[num_cols]), columns=num_cols)

# カテゴリ列の補完：最頻値
cat_imputer = SimpleImputer(strategy='most_frequent')
combined_cat = pd.DataFrame(cat_imputer.fit_transform(combined[cat_cols]), columns=cat_cols)

# 結合とエンコーディング
combined_imputed = pd.concat([combined_num, combined_cat], axis=1)
combined_encoded = pd.get_dummies(combined_imputed, columns=cat_cols, drop_first=True)

# 学習用と本番予測用に再分割
X_train_encoded = combined_encoded.iloc[:len(train_df)].copy()
X_test_encoded = combined_encoded.iloc[len(train_df):].copy()

# ==========================================================
# 3. モデルの内部評価（検証データでのスコア算出）
# ==========================================================
X_tr, X_val, y_tr, y_val = train_test_split(X_train_encoded, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

print("--- 勾配ブースティングモデルを構築中 ---")
# 💡 スコアを少しでも上げるため、パラメータをコンペ用に微調整しています
gb_model = GradientBoostingClassifier(
    n_estimators=150,       # 木の数を少し増やして表現力をアップ
    learning_rate=0.05,     # 学習率を小さくして、慎重に過学習を防ぎながら学習
    max_depth=4,            # 木の深さを制限して、細かすぎるノイズに反応させない
    min_samples_leaf=5,     # 末端の葉に必要な最小データ数を設定して過学習防止
    random_state=42
)

# 検証用データで学習と評価
gb_model.fit(X_tr, y_tr)
gb_probs = gb_model.predict_proba(X_val)[:, 1]
gb_auc = roc_auc_score(y_val, gb_probs)
print(f"■ 勾配ブースティング 検証用AUCスコア: {gb_auc:.4f}")

# ==========================================================
# 4. 本番予測と提出用ファイルの作成
# ==========================================================
print("\ntest.csv の Drafted（指名確率）を予測中...")
# 本番のテストデータに対して、ドラフトされる「確率（0〜1）」を予測
test_predictions_proba = gb_model.predict_proba(X_test_encoded)[:, 1]

# 結果をデータフレームにまとめる
submission = pd.DataFrame({
    'Id': test_ids,
    'Drafted': test_predictions_proba  # マイナスなし、きれいな0.0〜1.0の確率が出力されます
})

# CSVファイルとして出力
output_filename = 'submission_gradient_boosting.csv'
submission.to_csv(output_filename, index=False)
print(f"完了しました！ 予測結果は '{output_filename}' に保存されました。")
















































import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

# ==========================================================
# 【設定：ファイルのパス】
# ==========================================================
TRAIN_DATA_PATH = '/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/train.csv'  # 学習用ファイル名
TEST_DATA_PATH = '/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/test.csv'    # 予測用ファイル名

# ==========================================================
# 1. データの読み込み
# ==========================================================
print("データを読み込み中...")
train_df = pd.read_csv(TRAIN_DATA_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)

test_ids = test_df['Id']
y_train_full = train_df['Drafted'].astype(int)

# データの分離と一時的な結合（前処理のため）
X_train_raw = train_df.drop(columns=['Id', 'Drafted'], errors='ignore').copy()
X_test_raw = test_df.drop(columns=['Id', 'Drafted'], errors='ignore').copy()

# どちらのデータ由来かわかるようにフラグを立てて結合
X_train_raw['is_test'] = 0
X_test_raw['is_test'] = 1
combined = pd.concat([X_train_raw, X_test_raw], axis=0, ignore_index=True)

# ==========================================================
# 2. 工夫①：ポジションごとの欠損値補完
# ==========================================================
print("工夫①：ポジションごとの平均値・最頻値で補完中...")

# 補完対象の列を定義
num_cols = ['Age', 'Height', 'Weight', 'Sprint_40yd', 'Vertical_Jump', 'Bench_Press_Reps', 'Broad_Jump', 'Agility_3cone', 'Shuttle']
cat_cols = ['Player_Type', 'Position_Type'] # Position自体はグループ化に使うので除外

# ポジション（Position）ごとにループして、それぞれの平均・最頻値で埋める
# もしそのポジション全員が欠損している場合の保険として、全体の平均・最頻値も計算しておく
global_means = combined[num_cols].mean()
global_modes = combined[cat_cols].mode().iloc[0]

def impute_by_position(group):
    # 数値列の補完（ポジション別平均、無ければ全体平均）
    for col in num_cols:
        fill_val = group[col].mean()
        if pd.isna(fill_val):
            fill_val = global_means[col]
        group[col] = group[col].fillna(fill_val)
        
    # カテゴリ列の補完（ポジション別最頻値、無ければ全体最頻値）
    for col in cat_cols:
        if not group[col].dropna().empty:
            fill_val = group[col].mode().iloc[0]
        else:
            fill_val = global_modes[col]
        group[col] = group[col].fillna(fill_val)
        
    return group

# ポジションごとにグループ化して補完を実行
combined_imputed = combined.groupby('Position', group_keys=False).apply(impute_by_position)

# 一旦、学習データとテストデータに再分割
X_train_imp = combined_imputed[combined_imputed['is_test'] == 0].drop(columns=['is_test']).reset_index(drop=True)
X_test_imp = combined_imputed[combined_imputed['is_test'] == 1].drop(columns=['is_test']).reset_index(drop=True)

# ==========================================================
# 3. 工夫②：大学名（School）の Target Encoding
# ==========================================================
print("工夫②：大学名（School）の Target Encoding を実行中...")

# 過学習を防ぐためのスムージングパラメータ（過去のドラフト全体の平均にどれくらい引っ張るか）
smoothing = 10
global_mean_drafted = y_train_full.mean()

# 学習データへのTarget Encoding (K-Foldを使ってリークを防ぐ)
X_train_imp['School_TE'] = np.nan
kf = KFold(n_splits=5, shuffle=True, random_state=42)

for train_idx, val_idx in kf.split(X_train_imp):
    # 4/5のデータから、大学ごとの「指名件数」と「総数」を計算
    X_tr_fold = X_train_imp.iloc[train_idx]
    y_tr_fold = y_train_full.iloc[train_idx]
    
    stats = y_tr_fold.groupby(X_tr_fold['School']).agg(['count', 'mean'])
    
    # スムージングを適用した数式でTEを計算
    smoothed_te = (stats['count'] * stats['mean'] + smoothing * global_mean_drafted) / (stats['count'] + smoothing)
    
    # 残り1/5の検証用データにマッピング
    X_train_imp.loc[val_idx, 'School_TE'] = X_train_imp.loc[val_idx, 'School'].map(smoothed_te)

# 欠損（学習時に登場しなかった新しい大学など）は全体の平均値で埋める
X_train_imp['School_TE'] = X_train_imp['School_TE'].fillna(global_mean_drafted)

# テストデータへのTarget Encoding (学習データ全体の統計値を使う)
full_stats = y_train_full.groupby(X_train_imp['School']).agg(['count', 'mean'])
full_smoothed_te = (full_stats['count'] * full_stats['mean'] + smoothing * global_mean_drafted) / (full_stats['count'] + smoothing)
X_test_imp['School_TE'] = X_test_imp['School'].map(full_smoothed_te).fillna(global_mean_drafted)

# 役割を終えた元の School 列は削除
X_train_imp = X_train_imp.drop(columns=['School'])
X_test_imp = X_test_imp.drop(columns=['School'])

# ==========================================================
# 4. 残りのカテゴリ変数のワンホットエンコーディング
# ==========================================================
# 再び結合して get_dummies を適用
X_train_imp['is_test'] = 0
X_test_imp['is_test'] = 1
combined_final = pd.concat([X_train_imp, X_test_imp], axis=0, ignore_index=True)

# 残りのカテゴリ列（Player_Type, Position_Type, Position）をダミー化
combined_encoded = pd.get_dummies(combined_final, columns=['Player_Type', 'Position_Type', 'Position'], drop_first=True)

X_train_encoded = combined_encoded[combined_encoded['is_test'] == 0].drop(columns=['is_test']).reset_index(drop=True)
X_test_encoded = combined_encoded[combined_encoded['is_test'] == 1].drop(columns=['is_test']).reset_index(drop=True)

# ==========================================================
# 5. モデルの内部評価
# ==========================================================
X_tr, X_val, y_tr, y_val = train_test_split(X_train_encoded, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full)

print("\n--- 発展版モデルの検証中 ---")
# ランダムフォレスト
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_tr, y_tr)
rf_probs = rf_model.predict_proba(X_val)[:, 1]
rf_auc = roc_auc_score(y_val, rf_probs)
print(f"■ 発展版 ランダムフォレスト AUC: {rf_auc:.4f}")

# 勾配ブースティング
gb_model = GradientBoostingClassifier(
    n_estimators=150,       # 木の数を少し増やしてじっくり学習
    learning_rate=0.05,     # 学習スピードを少し遅くして慎重に（過学習防止）
    max_depth=4,            # 木の深さを制限（細かすぎるルールを作らせない）
    min_samples_leaf=5,     # 末端の葉に最低限必要なデータ数（ノイズ対策）
    random_state=42
)
gb_model.fit(X_tr, y_tr)
gb_probs = gb_model.predict_proba(X_val)[:, 1]
gb_auc = roc_auc_score(y_val, gb_probs)
print(f"■ 発展版 勾配ブースティング AUC: {gb_auc:.4f}")

# ==========================================================
# 6. 本番予測と保存
# ==========================================================
if gb_auc > rf_auc:
    print("\n→ 勾配ブースティングを採用して本番予測を行います。")
    final_model = gb_model
else:
    print("\n→ ランダムフォレストを採用して本番予測を行います。")
    final_model = rf_model

print("test.csv の Drafted（指名確率）を予測中...")
test_predictions_proba = final_model.predict_proba(X_test_encoded)[:, 1]

# 保存
submission = pd.DataFrame({
    'Id': test_ids,
    'Drafted': test_predictions_proba
})

output_filename = 'predicted_result_advanced.csv'
submission.to_csv(output_filename, index=False)
print(f"完了しました！ 予測結果は '{output_filename}' に保存されました。")