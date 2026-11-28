import pandas as pd
df=pd.read_csv("exercise_data_day14.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())


df["usage_minutes"]=df["usage_minutes"].fillna(df["usage_minutes"].mean())
df["churned"]=df["churned"].fillna(df["churned"].mean())


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error

from matplotlib import pyplot as plt
plt.scatter(df["usage_minutes"],df["support_tickets"])
#plt.show()
plt.scatter(df["churned"],df["usage_minutes"])
#plt.show()

x=df[["support_tickets","churned"]]
y=df["usage_minutes"]
model=LinearRegression()
kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print(scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print(mse)
print(r2)
print(model.coef_)