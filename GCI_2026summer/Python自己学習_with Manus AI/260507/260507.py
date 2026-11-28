import pandas as pd
df=pd.read_csv("data.csv")
print(df.head())
print(df.describe())
print(df.isnull().sum())


from matplotlib import pyplot as plt
df_categ=df.groupby(df["Category"])["Sales"].sum()
df_categ=df_categ.sort_values(ascending=False)
print(df_categ)


df["Date"]=pd.to_datetime(df["Date"])
df_monthly=df.groupby(df["Date"].dt.strftime("%Y-%m"))["Sales"].sum().reset_index()
df_monthly.columns=["Date","Total_Sales"]
print(df_monthly)
plt.bar(df_monthly["Date"],df_monthly["Total_Sales"])
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)



plt.show()


print("最も売上高が高かった月は一月で、その売り上げは19601である。")
