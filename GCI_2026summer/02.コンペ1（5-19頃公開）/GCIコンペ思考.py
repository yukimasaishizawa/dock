import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold,train_test_split
from sklearn.metrics import mean_squared_error,r2_score
df=pd.read_csv("/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/train.csv")
df=df.dropna()

categorical_cols = ["Player_Type", "Position_Type", "Position"]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=int)

columns_to_drop = ["Id","Drafted", "School"] 

existing_drop_cols = [col for col in columns_to_drop if col in df_encoded.columns]

x = df_encoded.drop(columns=existing_drop_cols) 
y = df_encoded["Drafted"]

model=LinearRegression()
kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print(scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(mse)
print(r2)

print(df["Year"].corr(df["Drafted"]))
print(df["Age"].corr(df["Drafted"]))
print(df["Height"].corr(df["Drafted"]))
print(df["Weight"].corr(df["Drafted"]))
print(df["Sprint_40yd"].corr(df["Drafted"]))
print(df["Vertical_Jump"].corr(df["Drafted"]))
print(df["Bench_Press_Reps"].corr(df["Drafted"]))
print(df["Broad_Jump"].corr(df["Drafted"]))
print(df["Agility_3cone"].corr(df["Drafted"]))
print(df["Shuttle"].corr(df["Drafted"]))




import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# 1. データの読み込み
train_path = "/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/train.csv"
test_path = "/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/test.csv"  # アップロードされたファイル名

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# 共通で使う数値列の定義
numeric_cols = [
    "Age",
    "Height",
    "Weight",
    "Sprint_40yd",
    "Vertical_Jump",
    "Bench_Press_Reps",
    "Broad_Jump",
    "Agility_3cone",
    "Shuttle",
]
categorical_cols = ["Player_Type", "Position_Type", "Position"]

# 2. 欠損値補完のための「基準値」をTrainデータから事前に計算
overall_medians = {col: train_df[col].median() for col in numeric_cols}
pos_medians = {
    col: train_df.groupby("Position")[col].median() for col in numeric_cols
}


# 3. 前処理と特徴量エンジニアリングを行う関数
def preprocess_pipeline(df, is_train=True):
    df_out = df.copy()

    # (1) 欠損値の補完
    for col in numeric_cols:
        if col in df_out.columns:
            # ポジション別中央値でマッピング
            pos_med = df_out["Position"].map(pos_medians[col])
            df_out[col] = df_out[col].fillna(pos_med)
            # それでも残る場合は全体中央値
            df_out[col] = df_out[col].fillna(overall_medians[col])

    # (2) 特徴量作成（BMI、体重×スピード、体重×瞬発力）
    df_out["BMI"] = df_out["Weight"] / (df_out["Height"] ** 2)
    df_out["Weight_Speed_Ratio"] = df_out["Weight"] / df_out["Sprint_40yd"]
    df_out["Weight_Juice"] = df_out["Weight"] * df_out["Vertical_Jump"]

    return df_out


# 先にTrainとTestの基礎的な前処理とBMI作成を終わらせる
train_proc = preprocess_pipeline(train_df, is_train=True)
test_proc = preprocess_pipeline(test_df, is_train=False)

# (3) ポジション内差分の計算（Trainで作成済みの特徴量を基準にするように変更）
target_cols = numeric_cols + ["BMI", "Weight_Speed_Ratio", "Weight_Juice"]
for col in target_cols:
    if col != "Age":
        # 必ず前処理済みの train_proc からポジション平均を算出する（エラー回避）
        train_pos_mean = train_proc.groupby("Position")[col].mean()
        global_mean = train_proc[col].mean()

        # Trainデータへの適用
        pos_mean_train = train_proc["Position"].map(train_pos_mean).fillna(global_mean)
        train_proc[f"{col}_diff_by_pos"] = train_proc[col] - pos_mean_train

        # Testデータへの適用（基準はTrainの平均）
        pos_mean_test = test_proc["Position"].map(train_pos_mean).fillna(global_mean)
        test_proc[f"{col}_diff_by_pos"] = test_proc[col] - pos_mean_test

# (4) 総合身体能力スコアの作成
positive_feats = [
    "Vertical_Jump_diff_by_pos",
    "Bench_Press_Reps_diff_by_pos",
    "Broad_Jump_diff_by_pos",
    "Weight_Juice_diff_by_pos",
]
negative_feats = [
    "Sprint_40yd_diff_by_pos",
    "Agility_3cone_diff_by_pos",
    "Shuttle_diff_by_pos",
]

train_proc["Athletic_Score"] = 0
test_proc["Athletic_Score"] = 0

for f in positive_feats:
    train_std = train_proc[f].std()
    denom = train_std if train_std != 0 else 1
    train_proc["Athletic_Score"] += train_proc[f] / denom
    test_proc["Athletic_Score"] += test_proc[f] / denom

for f in negative_feats:
    train_std = train_proc[f].std()
    denom = train_std if train_std != 0 else 1
    train_proc["Athletic_Score"] -= train_proc[f] / denom
    test_proc["Athletic_Score"] -= test_proc[f] / denom

# 「若さ × 身体能力」の追加
train_proc["Age_Athletic_Ratio"] = train_proc["Athletic_Score"] / (
    train_proc["Age"] - 18 + 1
)
train_proc["Age_Sprint_Prod"] = train_proc["Age"] * train_proc["Sprint_40yd"]

test_proc["Age_Athletic_Ratio"] = test_proc["Athletic_Score"] / (
    test_proc["Age"] - 18 + 1
)
test_proc["Age_Sprint_Prod"] = test_proc["Age"] * test_proc["Sprint_40yd"]


# 4. スムージング付き大学ドラフト率の計算と適用
global_draft_mean = train_proc["Drafted"].mean()
m = 15
stats = train_proc.groupby("School")["Drafted"].agg(["count", "mean"])
school_rates = (stats["count"] * stats["mean"] + m * global_draft_mean) / (
    stats["count"] + m
)

train_proc["School_Draft_Rate"] = (
    train_proc["School"].map(school_rates).fillna(global_draft_mean)
)
test_proc["School_Draft_Rate"] = (
    test_proc["School"].map(school_rates).fillna(global_draft_mean)
)

# 5. 特徴量とターゲットの分離
y_train = train_proc["Drafted"].astype(int)
X_train_raw = train_proc.drop(columns=["Id", "School", "Drafted"])
X_test_raw = test_proc.drop(columns=["Id", "School"], errors="ignore")

# カテゴリ変数のOne-Hot Encoding（TrainとTestを結合して列のズレを完全に防ぐ）
X_all = pd.concat([X_train_raw, X_test_raw], axis=0)
X_all_encoded = pd.get_dummies(
    X_all, columns=categorical_cols, drop_first=True, dtype=int
)

X_train_final = X_all_encoded.iloc[: len(train_df)]
X_test_final = X_all_encoded.iloc[len(train_df) :]

# 6. 全データを使って最終モデルの学習
model_gb = GradientBoostingClassifier(
    n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42
)
model_rf = RandomForestClassifier(
    n_estimators=150, max_depth=8, min_samples_leaf=5, random_state=42, n_jobs=-1
)

model_gb.fit(X_train_final, y_train)
model_rf.fit(X_train_final, y_train)

# 7. テストデータの予測（確率を小数で出力し、アンサンブル）
pred_gb = model_gb.predict_proba(X_test_final)[:, 1]
pred_rf = model_rf.predict_proba(X_test_final)[:, 1]

# 勾配ブースティングとランダムフォレストのブレンド
final_predictions = (pred_gb * 0.8) + (pred_rf * 0.2)

# 8. コンペ提出用ファイルの作成
submission = pd.DataFrame({"Id": test_df["Id"], "Drafted": final_predictions})

# 保存
submission.to_csv("submission.csv", index=False)
print("提出用ファイル 'submission.csv' が正常に作成されました！")

# 中身の確認表示
print(submission.head())