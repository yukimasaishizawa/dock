import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler as sc

df=pd.read_csv("exercise_data_day25.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df["category"]=df["category"].str.lower()
df["region"]=df["region"].str.lower()

quantity=df["quantity"].mode()[0]
df["quantity"]=df["quantity"].fillna(quantity)
region_dropna=df["region"].dropna()
region_mode=region_dropna.mode()[0]
df["region"]=df["region"].fillna(region_mode)
print(df.isnull().sum())

region=pd.get_dummies(df["region"],dtype=int,drop_first=False)    #set drop first as False because if set True,"central" will delet from df
categ=pd.get_dummies(df["category"],dtype=int,drop_first=False)   #set drop first as False because if set True,"beuty" will delet from df
df=pd.concat([df,categ,region],axis=1)
df["sales_volume"]=(df["price"]*df["quantity"])
print(df.head())
print(df["category"].unique())
print(df["region"].unique())


x=df[["price",'clothing', 'beauty', 'books' ,'home' ,'electronics','north', 'east' ,'west', 'central', 'south']]
y=df["sales_volume"]
model=LinearRegression()
kf=KFold(n_splits=4,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sc=sc()
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


