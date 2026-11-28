import pandas as pd
df=pd.read_csv("sales_data.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error

#plt.scatter(df["Ad_Spend"],df["Sales"])
#plt.show()
#plt.scatter(df["Promo_Spend"],df["Sales"])
#plt.show()

model=LinearRegression()
kf=KFold(n_splits=5,shuffle=True,random_state=42)
x=df[["Ad_Spend","Promo_Spend"]]
y=df["Sales"]
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print(scores)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_pred,y_test)
r2=r2_score(y_pred,y_test)
print("mean_squared_error is",mse)
print("r2 score is",r2)

x_new=pd.DataFrame([[800,400]],columns=["Ad_Spend","Promo_Spend"])
prediction=model.predict(x_new)
print("予測結果;",prediction[0])


