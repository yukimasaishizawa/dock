import pandas as pd
df=pd.read_csv("exercise_data_day1.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df["stock_level"]=df["stock_level"].fillna(df["stock_level"].mean())

print(df.isnull().sum())

print(df["date"].corr(df["stock_level"]))
print(df["sku"].corr(df["stock_level"]))
print(df["incoming_shipment"].corr(df["stock_level"]))
print(df["sales_volume"].corr(df["stock_level"]))

from matplotlib import pyplot as plt
import numpy as np
x=df["sku"]
y=df["stock_level"]

coeffs=np.polyfit(x,y,1)
poly_func=np.poly1d(coeffs)
x_fit=np.linspace(x.min(),x.max(),100)
y_fit=poly_func(x_fit)
plt.figure(figsize=(20,10))
plt.plot(x_fit,y_fit,color="red",label=str(poly_func))
plt.scatter(x,y)
plt.legend()
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error

x2=df[["date","sku","incoming_shipment","sales_volume"]]
y2=df["stock_level"]
model=LinearRegression()
kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,x2,y2,cv=kf,scoring="r2")
print("cross_val_score:",scores)

x_train,x_test,y_train,y_test=train_test_split(x2,y2,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_error:",mse)
print("r2_score:",r2)










