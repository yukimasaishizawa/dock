import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error,r2_score

df=pd.read_csv("exercise_data_day20.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df2=df
df2=df2.dropna()
df2["sales_volume"]=(df2["price"]*df2["quantity"])
df["sales_volume"]=(df["price"]*df["quantity"])
df["category"]=df["category"].str.lower()
print(df.head())
region=pd.get_dummies(df["region"],dtype=int)
categ=pd.get_dummies(df["category"],dtype=int)
df=pd.concat([df,region,categ],axis=1)
region2=pd.get_dummies(df2["region"],dtype=int)
categ2=pd.get_dummies(df2["category"],dtype=int)
df2=pd.concat([df2,region2,categ2],axis=1)
print(df.head())
print(df2.isnull().sum())
print(df.describe())
print(df2.head())
print(df["category"].unique())
print(df["region"].unique())

print("-------include categ,price,region--------")

x=df[['beauty', 'electronics', 'clothing', 'books', 'home', "price", 'North', 'West' ,'Central', 'South', 'East']]
y=df["sales_volume"]
model=LinearRegression()
kf=KFold(n_splits=4,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_error:",mse)
print("r2_score:",r2)
print("coef:",model.coef_)

print("-------include price,region--------")

x=df[["price", 'North', 'West' ,'Central', 'South', 'East']]
y=df["sales_volume"]
model=LinearRegression()
kf=KFold(n_splits=4,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_error:",mse)
print("r2_score:",r2)
print("coef:",model.coef_)

print("-------include categ,price,region,exclude NaN--------")

x=df2[['beauty', 'electronics', 'clothing', 'books', 'home', "price", 'North', 'West' ,'Central', 'South', 'East']]
y=df2["sales_volume"]
model=LinearRegression()
kf=KFold(n_splits=4,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_error:",mse)
print("r2_score:",r2)
print("coef:",model.coef_)

print("-------include price,region,exclude NaN--------")

x=df2[["price", 'North', 'West' ,'Central', 'South', 'East']]
y=df2["sales_volume"]
model=LinearRegression()
kf=KFold(n_splits=4,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_error:",mse)
print("r2_score:",r2)
print("coef:",model.coef_)