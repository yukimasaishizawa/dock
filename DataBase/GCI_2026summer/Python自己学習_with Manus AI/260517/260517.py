import pandas as pd
df=pd.read_csv("exercise_data_day2.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df["support_tickets"]=df["support_tickets"].fillna(df["support_tickets"].mean())

print("corriration between support_tickets and churned is",df["support_tickets"].corr(df["churned"]))

from matplotlib import pyplot as plt
import numpy as np


x=df["support_tickets"]
y=df["churned"]
coeffs=np.polyfit(x,y,1)
poly_func=np.poly1d(coeffs)
x_fit=np.linspace(x.min(),x.max(),100)
y_fit=poly_func(x_fit)


plt.figure(figsize=(20,10))
plt.scatter(df["support_tickets"],df["churned"])
plt.xlabel("suport_tickets")
plt.ylabel("churned")
plt.plot(x_fit,y_fit,color="red",label=str(poly_func))
plt.legend()
plt.show()


from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

x=df[["support_tickets"]]
y=df["churned"]
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



