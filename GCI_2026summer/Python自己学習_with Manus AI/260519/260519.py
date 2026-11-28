import pandas as pd
df=pd.read_csv("exercise_data_day4.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())


df["pm25"]=df["pm25"].fillna(df["pm25"].mean())
df["temperature"]=df["temperature"].fillna(df["temperature"].mean())

df.insert(4,"temp*hum",(df["temperature"]*df["humidity"]))


from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error

plt.scatter((df["temperature"]*df["humidity"]),df["pm25"])
#plt.show()

#pm25の値に対し、no2が関係あるか調べる
print(df["pm25"].corr(df["no2"]))
print(df["no2"].corr(df["pm25"]))
#関係はなさそうである。

x=df[["temperature","humidity"]]
y=df["pm25"]
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



x=df[["temp*hum"]]
y=df["pm25"]
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








