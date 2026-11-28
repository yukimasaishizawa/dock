import pandas as pd
from pycaret.regression import *

# 1. データの読み込み
# ※ 提出用のワインデータか、演習用のデータ（exercise_data_day15）か、使いたい方を指定してください
df = pd.read_csv("exercise_data_day21.csv")

df["usage_minutes"]=df["usage_minutes"].fillna(df["usage_minutes"].mean())
df["support_tickets"]=df["support_tickets"].fillna(df["support_tickets"].mean())

df=df[["usage_minutes","support_tickets"]]
# 2. データの確認（欠損値などがないか最初に見る）
print("--- データ確認 ---")
print(df.isnull().sum())

# 3. PyCaretの環境セットアップ
# target には予測したい「目的変数」の列名を入れてください（例: 'quality' など）
exp_reg = setup(data=df, target='usage_minutes', session_id=123, verbose=True)

# 4. 全モデルを一括比較！（画面にランキング表が出ます） ⭐️ここがやりたいこと！
best_model = compare_models()

# 5. 一番良かったモデルの詳細を確認する
print("\n--- 最も優秀だったモデルのパラメータ ---")
print(best_model)