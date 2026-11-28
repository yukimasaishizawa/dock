import pandas as pd
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("exercise_data_day28.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())
print(df["category"].unique())

product=df["product_id"].mode()[0]
df["product_id"]=df["product_id"].fillna(product)
df["category"]=df["category"].str.lower()
categ=df["category"].mode()[0]
df["category"]=df["category"].fillna(categ)
df["quantity"]=df["quantity"].fillna(df["quantity"].mode()[0])
print(df.isnull().sum())
print(df.head())
print(df["product_id"].unique())

print(df["category"].unique())
print(df["region"].unique())

df_categ=pd.get_dummies(df["category"],dtype=int)
df_region=pd.get_dummies(df["region"],dtype=int)
df_id=pd.get_dummies(df["product_id"],dtype=int)
df=pd.concat([df[["price","quantity"]],df_categ,df_region,df_id],axis=1)
print(df.head())


x=df[["quantity",'books', 'home', 'electronics', 'clothing', 'beauty','West', 'East', 'Central' ,'South', 'North']]
y=df["price"]
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

