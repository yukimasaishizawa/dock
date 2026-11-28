import pandas as pd
df=pd.read_csv("sales_data.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())


from sklearn.linear_model import LinearRegression
from matplotlib import pylab as plt
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error

model=LinearRegression()
x=df[["面積","従業員数","競合店舗数","プロモーション費用"]]
y=df["月間売上"]
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


plt.scatter(df["面積"],df["月間売上"])
plt.show()

plt.scatter(df["従業員数"],df["月間売上"])
plt.show()

plt.scatter(df["競合店舗数"],df["月間売上"])
plt.show()

plt.scatter(df["プロモーション費用"],df["月間売上"])
plt.show()