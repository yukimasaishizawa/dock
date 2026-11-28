import pandas as pd
df_train=pd.read_csv("/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/train.csv")
print(df_train.head())
print(df_train.isnull().sum())
print(df_train.describe())



#print(df_dropna["Year"].corr(df_dropna["Drafted"]))
#print(df_dropna["Age"].corr(df_dropna["Drafted"]))
#print(df_dropna["School"].corr(df_dropna["Drafted"]))
#print(df_dropna["Height"].corr(df_dropna["Drafted"]))
#print(df_dropna["Weight"].corr(df_dropna["Drafted"]))
#print(df_dropna["Sprint_40yd"].corr(df_dropna["Drafted"]))
#print(df_dropna["Vertical_Jump"].corr(df_dropna["Drafted"]))
#print(df_dropna["Bench_Press_Reps"].corr(df_dropna["Drafted"]))
#print(df_dropna["Broad_Jump"].corr(df_dropna["Drafted"]))
#print(df_dropna["Agility_3cone"].corr(df_dropna["Drafted"]))
#print(df_dropna["Shuttle"].corr(df_dropna["Drafted"]))
#print(df_dropna["Player_Type"].corr(df_dropna["Drafted"]))
#print(df_dropna["Position_Type"].corr(df_dropna["Drafted"]))
#print(df_dropna["Position "].corr(df_dropna["Drafted"]))



from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error,r2_score


df_train["Age"] = df_train["Age"].fillna(df_train["Age"].mean())
df_train["Sprint_40yd"] = df_train["Sprint_40yd"].fillna(df_train["Sprint_40yd"].mean())
df_train["Vertical_Jump"] = df_train["Vertical_Jump"].fillna(df_train["Vertical_Jump"].mean())
df_train["Bench_Press_Reps"] = df_train["Bench_Press_Reps"].fillna(df_train["Bench_Press_Reps"].mean())
df_train["Broad_Jump"] = df_train["Broad_Jump"].fillna(df_train["Broad_Jump"].mean())
df_train["Agility_3cone"] = df_train["Agility_3cone"].fillna(df_train["Agility_3cone"].mean())
df_train["Shuttle"] = df_train["Shuttle"].fillna(df_train["Shuttle"].mean())

categorical_cols = ["Player_Type", "Position_Type", "Position"]
df_encoded = pd.get_dummies(df_train, columns=categorical_cols, drop_first=True, dtype=int)

columns_to_drop = ["Id","Drafted", "School"] 

existing_drop_cols = [col for col in columns_to_drop if col in df_encoded.columns]

x = df_encoded.drop(columns=existing_drop_cols) 
y = df_encoded["Drafted"]

remaining_strings = x.select_dtypes(include=['object']).columns
print("まだ残ってしまっている文字列の列:", remaining_strings)

model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, x, y, cv=kf, scoring="r2")
print("cross_val_score:", scores)
print("平均 R2 スコア:", scores.mean())

model.fit(x, y)
df_test=pd.read_csv("/Users/yukimasaishizawa/Documents/GCI/02.コンペ1（5-19頃公開）/data/test.csv")

for col in df_test.columns:
    if df_test[col].isnull().any():
        if col in df_train.columns and pd.api.types.is_numeric_dtype(df_train[col]):
            df_test[col] = df_test[col].fillna(df_train[col].mean())
        elif pd.api.types.is_numeric_dtype(df_test[col]):
            df_test[col] = df_test[col].fillna(df_test[col].mean())

df_test_processed = df_test.copy()
categorical_cols = ["Player_Type", "Position_Type", "Position"]
df_test_encoded = pd.get_dummies(df_test_processed, columns=categorical_cols, drop_first=True, dtype=int)

x_test = df_test_encoded.reindex(columns=x.columns, fill_value=0)

nan_cols = x_test.columns[x_test.isnull().any()].tolist()
print("まだ NaN が残っている列:", nan_cols)

test_predictions = model.predict(x_test)

submission_df = pd.DataFrame({
    "Id": df_test["Id"], 
    "Drafted": test_predictions
})

submission_df.to_csv("predicted.csv", index=False)
print("predicted.csv が保存されました！")


