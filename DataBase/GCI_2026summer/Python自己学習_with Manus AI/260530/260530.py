import pandas as pd
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from matplotlib import pyplot as plt

df=pd.read_csv("exercise_data_day15.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df["price"]=df["price"].fillna(df["price"].mean())
df["quantity"]=df["quantity"].astype("Int64")
df["quantity"]=df["quantity"].fillna(df["quantity"].mode()[0])

print(df.isnull().sum())


df["sales_volume"]=(df["price"]*df["quantity"])


region=pd.get_dummies(df["region"],dtype=int,drop_first=False)
df=pd.concat([df,region],axis=1)
print(df.head())

model=LinearRegression()
x=df[["price","Central","East","North","South","West"]]
y=df["sales_volume"]
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