import pandas as pd
df=pd.read_csv("exercise_data.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df["Sales"]=df["Sales"].fillna(df["Sales"].mean())
df["Customers"]=df["Customers"].fillna(df["Customers"].mean())
df["Open_Hours"]=df["Open_Hours"].fillna(df["Open_Hours"].mean())
df["Promotion"]=df["Promotion"].fillna(df["Promotion"].mean())
df["Holiday"]=df["Holiday"].fillna(df["Holiday"].mean())

print(df.isnull().sum())

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error

cust_corr=df["Sales"].corr(df["Customers"])
open_corr=df["Sales"].corr(df["Open_Hours"])
prom_corr=df["Sales"].corr(df["Promotion"])
hold_corr=df["Sales"].corr(df["Holiday"])


print(cust_corr)
print(open_corr)
print(prom_corr)
print(hold_corr)

print("Open_Hours han strongest relationship to Sales")


x=df[["Customers","Open_Hours","Promotion","Holiday"]]
y=df["Sales"]
model=LinearRegression()
kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
print("r2_score:",r2)
print("mean_squared_error:",mse)
