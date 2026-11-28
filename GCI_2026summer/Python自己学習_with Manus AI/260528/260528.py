import pandas as pd
df=pd.read_csv("exercise_data_day13.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())


df["sales_volume"]=df["sales_volume"].fillna(df["sales_volume"].mean())
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from matplotlib import pyplot as plt

x=df[["stock_level","incoming_shipment"]]
y=df["sales_volume"]
model=LinearRegression()
kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_error:",mse)
print("r2_score:",r2)
print(model.coef_)
