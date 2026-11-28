import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from matplotlib import pyplot as plt

df=pd.read_csv("exercise_data_day18.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df["pm25"]=df["pm25"].fillna(df["pm25"].mean())
df["no2"]=df["no2"].fillna(df["no2"].mean())
df["temperature"]=df["temperature"].fillna(df["temperature"].mean())
df["humidity"]=df["humidity"].fillna(df["humidity"].mean())
print(df.isnull().sum())

plt.scatter(df["pm25"],df["no2"])
#plt.show()
print(df["pm25"].corr(df["no2"]))
print(df["no2"].corr(df["temperature"]))
print(df["no2"].corr(df["humidity"]))




print("-------with temperature,humidity--------")
model=LinearRegression()
x=df[["temperature","humidity"]]
y=df["pm25"]
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
print("corf",model.coef_)


print("-------with temperature,humidity,no2--------")
model=LinearRegression()
x=df[["temperature","humidity","no2"]]
y=df["pm25"]
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
print("corf",model.coef_)