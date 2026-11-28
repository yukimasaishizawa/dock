import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

df=pd.read_csv("exercise_data_day12.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

weather_mode=df["weather_condition"].mode()[0]
df["weather_condition"]=df["weather_condition"].fillna(weather_mode)


df_weather=pd.get_dummies(df["weather_condition"],dtype=int)
df=pd.concat([df,df_weather],axis=1)
print(df.head())

print(df.isnull().sum())

x=df[["Clear","Cloudy","Rainy","Snowy"]]
y=df["vehicle_count"]
y2=df["average_speed"]


print("---------predict vehicle_count based on weather condition----------")
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



print("---------predict avarage_speed based on weather condition----------")
model2=LinearRegression()
kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model2,x,y2,cv=kf,scoring="r2")
print("cross_val_score:",scores)
x_train,x_test,y_train,y_test=train_test_split(x,y2,test_size=0.2,random_state=42)
model2.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_error:",mse)
print("r2_score:",r2)
print("coef:",model2.coef_)























