import pandas as pd
df=pd.read_csv("sales_data.csv")
print(df.head(10))
print(df.describe())
print(df.isnull().sum())

df["Date"]=pd.to_datetime(df["Date"])

df_daily=df.groupby(df["Date"].dt.to_period("D"))["Total_Sales"].sum()
df_daily=df_daily.sort_values(ascending=False)
df_daily = df.groupby(df["Date"].dt.date)["Total_Sales"].sum()
print(df_daily)

df_sort=df.sort_values(by="Total_Sales", ascending=False)
print(df_sort.head(10))

from matplotlib import pyplot as plt
plt.bar(df_daily.index, df_daily.values)
plt.xlabel("Date")
plt.ylabel("Total_Sales")
plt.title("Daily Total Sales")
#plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error, r2_score

model1=LinearRegression()
x=df[["Price"]]
y=df["Quantity"]
kf=KFold(n_splits=5, shuffle=True, random_state=42)
scores=cross_val_score(model1,x,y,cv=kf,scoring="r2")
print(scores)

model2=LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model2.fit(X_train,y_train)
y_pred=model2.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)










