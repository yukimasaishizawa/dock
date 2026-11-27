import pandas as pd
df=pd.read_csv("prog3-data-jp.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())


from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from matplotlib import pyplot as plt


plt.plot(df["日付"], df["根室"], marker='o', linestyle='-')
plt.xlabel("日付")
plt.ylabel("根室の気温")
plt.title("根室の気温の推移")
plt.xticks(rotation=45)
##plt.show()

plt.plot(df["日付"], df["東京"], marker='o', linestyle='-')
plt.xlabel("日付")
plt.ylabel("東京の気温")
plt.title("東京の気温の推移")
plt.xticks(rotation=45)
#plt.show()


plt.plot(df["日付"], df["長野"], marker='o', linestyle='-')
plt.xlabel("日付")
plt.ylabel("長野の気温")
plt.title("長野の気温の推移")
plt.xticks(rotation=45)
#plt.show()



# 実装イメージ
X = df[["根室"]] # 説明変数
y = df["東京"]   # 目的変数

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 予測と評価
y_pred = model.predict(X_test)
print(f"R2スコア: {r2_score(y_test, y_pred)}")

print(df["東京"])

mean_values=df[["東京","根室", "長野"]].mean(axis=0)
mean_row=mean_values.to_dict()
mean_row["日付"]="平均"
df2=pd.DataFrame([mean_row])
print(mean_row)
print(df2)

df_tokyo=df["東京"].mean()
print(df_tokyo)

df_tokyo=df["根室"].mean()
print(df_tokyo)

df_tokyo=df["長野"].mean()
print(df_tokyo)

dfNew=pd.concat([df, df2], ignore_index=True).round(1)
print(dfNew)

dfNew.to_csv("Combined_Data.csv", index=False)