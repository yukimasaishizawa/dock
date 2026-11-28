import pandas as pd
df=pd.read_csv("cafe_sales_data.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error

plt.scatter(df["temperature"],df["customers"])
plt.xlabel("temp")
plt.ylabel("cust")
#plt.show()

plt.bar(df["is_weekend"],df["customers"])
plt.xlabel("is weekend")
plt.ylabel("cust")
#plt.show()


model=LinearRegression()
x=df[["temperature"]]
y=df["customers"]

kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_error",mse)
print("r2_score",r2)


x_new=pd.DataFrame([30.0],columns=["temperature"])
prediction=model.predict(x_new)
print(prediction[0])




model2=LinearRegression()
x2=df[["temperature","is_weekend"]]
y2=df["customers"]

kf2=KFold(n_splits=5,shuffle=True,random_state=42)
scores2=cross_val_score(model2,x2,y2,cv=kf2,scoring="r2")
print("cross_val_score:",scores2)

x_train2,x_test2,y_train2,y_test2=train_test_split(x2,y2,test_size=0.2,random_state=42)
model2.fit(x_train2,y_train2)
y_pred2=model2.predict(x_test2)
mse2=mean_squared_error(y_test2,y_pred2)
r22=r2_score(y_test2,y_pred2)
print("mean_squared_error",mse2)
print("r2_score",r22)


x_new2=pd.DataFrame([[30.0,1]],columns=["temperature","is_weekend"])
prediction2=model2.predict(x_new2)
print(prediction2[0])






