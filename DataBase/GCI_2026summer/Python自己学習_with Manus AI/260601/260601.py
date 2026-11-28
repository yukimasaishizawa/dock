import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from matplotlib import pyplot as plt

df=pd.read_csv("exercise_data_day17.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df["stock_level"]=df["stock_level"].fillna(df["stock_level"].mean())
df["incoming_shipment"]=df["incoming_shipment"].fillna(df["incoming_shipment"].mean())

print(df.max()) #標準化の必要はなさそうである

x=df[["incoming_shipment","stock_level"]]
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
print("coef",model.coef_)
