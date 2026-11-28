import pandas as pd
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("exercise_data_day24.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df["sales_volume"]=df["sales_volume"].fillna(df["sales_volume"].mean())
print(df.isnull().sum())

print(df["sku"].corr(df["sales_volume"]))
print("There is almost no correlation between sku and sales_volume")

x=df[["stock_level","incoming_shipment"]]
y=df["sales_volume"]
model=LinearRegression()
kf=KFold(n_splits=4,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sc=StandardScaler()
sc.fit(x_train)
x_train_std=sc.transform(x_train)
x_test_std=sc.transform(x_test)
model.fit(x_train_std,y_train)
y_pred=model.predict(x_test_std)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_error:",mse)
print("r2_score:",r2)
print("coef",model.coef_)
print("if stock_level rise vy 1, then sales_volume will drop by 0.0066088")
print("if incoming_shipment rise vy 1, then sales_volume will rise by 0.00129628")