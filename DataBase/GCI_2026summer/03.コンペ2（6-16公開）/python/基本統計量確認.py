import pandas as pd
df=pd.read_csv("/Users/yukimasaishizawa/Documents/GCI/03.コンペ2（6-16公開）/python/train.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

print(df["annual_income"].unique())
df2=df["annual_income"]
print(df2.head())

df["annual_income"]=df["annual_income"].fillna(df["annual_income"].mean())
df.to_csv("train_mean.csv")
df_mode=df["annual_income"].mode()[0]
df["annual_income"]=df["annual_income"].fillna(df_mode)
print(df.isnull().sum())
df.to_csv("train_mode.csv")
df["annual_income"]=df["annual_income"].fillna(df["annual_income"].median())
df.to_csv("train_median.csv")