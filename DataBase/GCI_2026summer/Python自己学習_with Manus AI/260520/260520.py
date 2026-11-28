import pandas as pd
df=pd.read_csv("exercise_data_day5.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

print(df["weather_condition"].unique())

weather_map = {
    'Snowy': 0,
    'Clear': 1,
    'Rainy': 2,
    'Cloudy': 3
}

df["weather_condition"] = df["weather_condition"].map(weather_map)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,KFold,cross_val_score
from sklearn.metrics import mean_squared_error,r2_score

x=df[["average_speed","weather_condition"]]
y=df["vehicle_count"]

model=LinearRegression()
kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("R2 Scores:",scores)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("R2 Score:",r2)
