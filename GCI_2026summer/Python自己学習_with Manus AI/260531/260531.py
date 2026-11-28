import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from matplotlib import pyplot as plt

df=pd.read_csv("exercise_data_day16.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

print(type(df["vehicle_count"][1]))
df["vehicle_count"]=df["vehicle_count"].astype("Int64")
df["vehicle_count"]=df["vehicle_count"].fillna(df["vehicle_count"].mode()[0])
df["average_speed"]=df["average_speed"].fillna(df["average_speed"].mean())

weather=pd.get_dummies(df["weather_condition"],dtype=int)
df=pd.concat([df,weather],axis=1)

speed_max=df["average_speed"].max()
df["average_speed"]=(df["average_speed"]/speed_max)

print(df.head())
print(df.isnull().sum())

x=df[["average_speed","Clear","Cloudy","Rainy","Snowy"]]
y=df["vehicle_count"]
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
print("coef:",model.coef_)

