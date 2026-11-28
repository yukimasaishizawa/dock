import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("exercise_data_day22.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

v=df["vehicle_count"].mode()[0]
df["vehicle_count"]=df["vehicle_count"].fillna(v)
df["vehicle_count"]=df["vehicle_count"].astype(int)
print(df.isnull().sum())

weather=pd.get_dummies(df["weather_condition"],dtype=int)
df=pd.concat([df,weather],axis=1)
print(df.head())

from sklearn.preprocessing import StandardScaler
x=df[["average_speed","Clear","Cloudy","Rainy","Snowy"]]
y=df["vehicle_count"]
model=LinearRegression()
kf=KFold(n_splits=4,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
sc=StandardScaler()
sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)
model.fit(x_train_std,y_train)
y_pred=model.predict(x_test_std)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_error:",mse)
print("r2_score:",r2)
print("coef",model.coef_)






