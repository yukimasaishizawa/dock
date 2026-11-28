import pandas as pd
df=pd.read_csv("sales_data.csv")
df.insert(6,"Sales",df["Quantity"]*df["UnitPrice"])
print(df.head())
print(df.isnull().sum())
print(df.describe())

print("total sales:",(df["Quantity"]*df["UnitPrice"]).sum())


df["OrderDate"]=pd.to_datetime(df["OrderDate"])
df_monthly=df.groupby(df["OrderDate"].dt.to_period("M"))["Sales"].sum()
print(df_monthly)
from matplotlib import pyplot as plt
plt.bar(df_monthly.index.astype(str),df_monthly)
plt.xticks(rotation=45)
plt.show()


df_categ=df.groupby(df["Category"])["Sales"].sum()
df_categ=df_categ.sort_values(ascending=False)
print(df_categ)

print("Average sales:",df["Sales"].mean())