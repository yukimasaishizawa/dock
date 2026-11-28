import pandas as pd
df=pd.read_csv("exercise_data_day8.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

print(df["region"].unique())

df_dummis=pd.get_dummies(df["region"],dtype=int,drop_first=False)

df=pd.concat([df,df_dummis],axis=1)

df_sales=pd.DataFrame((df["price"]*df["quantity"]),columns=["sales_volume"])

df=pd.concat([df,df_sales],axis=1)

print(df.head())


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error

x=df[["price","Central","East","North","South","West"]]
y=df["sales_volume"]
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



print(model.coef_)