import pandas as pd
df=pd.read_csv("coffee_shop_sales.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df["日付"]=pd.to_datetime(df["日付"])

df_monthly=df.groupby(df["日付"].dt.to_period("M"))["売上"].sum()
print(df_monthly)

df_store_rank=df.groupby(df["店舗名"])["売上"].sum()
df_store_rank=df_store_rank.sort_values(ascending=False)
print(df_store_rank)

df_goods_rank=df.groupby(df["商品名"])["売上"].sum()
df_goods_rank=df_goods_rank.sort_values(ascending=False)
print(df_goods_rank)

df_daily=df.groupby(df["日付"].dt.to_period("D"))["売上"].sum()
print(df_daily)

#from matplotlib import pyplot as plt
#plt.scatter(df["商品名"],df["数量"])
#plt.xlabel="x"
#plt.ylabel="y"

#plt.show()


#plt.bar(df_store_rank.index,df_store_rank.values)
#plt.xlabel="店舗名"
#plt.ylabel="売上"
#plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
model1=LinearRegression()
x=df[["単価"]]
y=df[["数量"]]
kf=KFold(n_splits=5, shuffle=True, random_state=42)
scores1=cross_val_score(model1,x,y,cv=kf,scoring="r2")
print(scores1)


from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

model2=LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model2.fit(X_train,y_train)
y_pred=model2.predict(X_test)
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)

print(r2)
print(mse)


