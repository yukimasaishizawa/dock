import pandas as pd
df=pd.read_csv("user_behavior_data.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error

df["user_id"]=df["user_id"].astype(int)

mode_event=df["event_type"].mode()[0]
df["event_type"].fillna(mode_event)
df["product_id"].fillna(0)
df["event_type"]=df["event_type"].str.lower()
df["event_timestamp"]=pd.to_datetime(df["event_timestamp"])
event_group=df.groupby("event_type").size().sort_values(ascending=False)
print(event_group)

id_group=df.groupby("product_id").size().sort_values(ascending=False)
print(id_group)

print("もっと購入された製品",id_group.head(1))





