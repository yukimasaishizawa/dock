import pandas as pd
df=pd.read_csv("coffee_sales_data.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df_corr=df["temperature"].corr(df["sales_amount"])
print("corretion between temp and sales_amount is:",df_corr)

from matplotlib import pyplot as plt
plt.scatter(df["temperature"],df["sales_amount"])
plt.xlabel("temp")
plt.ylabel("sales_amount")
#plt.show()

from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

x=df[["temperature","is_weekend"]]
y=df["sales_amount"]
model=LinearRegression()
kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score",scores)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_error",mse)
print("r2_score",r2)

x_new=pd.DataFrame([[28,0]],columns=["temperature","is_weekend"])
prediction=model.predict(x_new)
print(prediction[0])






