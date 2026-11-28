import pandas as pd
df=pd.read_csv("exercise_data_day9.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

#fill Nan as average
df["pm25"]=df["pm25"].fillna(df["pm25"].mean())
df["temperature"]=df["temperature"].fillna(df["temperature"].mean())
df["humidity"]=df["humidity"].fillna(df["humidity"].mean())


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error

#analysis pm25 based on temperature and humidity
x=df[["temperature","humidity"]]
y=df["pm25"]
model=LinearRegression()
kf=KFold(n_splits=5,random_state=42,shuffle=True)
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


#delet Nan
df2=pd.read_csv("exercise_data_day9.csv")
df2=df2.dropna()

#analysis pm25 based on temperature and humidity
x=df2[["temperature","humidity"]]
y=df2["pm25"]
model=LinearRegression()
kf=KFold(n_splits=5,random_state=42,shuffle=True)
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


#analysis pm25 based on temperature,humidity and no2
x=df[["temperature","humidity","no2"]]
y=df["pm25"]
model=LinearRegression()
kf=KFold(n_splits=5,random_state=42,shuffle=True)
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


#analysis pm25 based on temperature,humidity and no2
x=df2[["temperature","humidity","no2"]]
y=df2["pm25"]
model=LinearRegression()
kf=KFold(n_splits=5,random_state=42,shuffle=True)
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
