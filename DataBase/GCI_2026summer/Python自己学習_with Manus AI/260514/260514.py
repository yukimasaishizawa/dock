import pandas as pd
df=pd.read_csv("sales_data.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error

plt.scatter(df["advertising_cost"],df["sales"])
plt.xlabel("ad_cost")
plt.ylabel("sales")
#plt.show()


x=df[["advertising_cost"]]
y=df["sales"]
model=LinearRegression()
kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_errore:",mse)
print("r2_score::",r2)


ad_cost=100
x_new=pd.DataFrame([[ad_cost]],columns=["advertising_cost"])
prediction=model.predict(x_new)
print(ad_cost,"万円の時の売り上げ予測:",prediction[0])











