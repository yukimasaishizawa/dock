import pandas as pd
df=pd.read_csv("exercise_data_day3.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df["quantity"]=df["quantity"].fillna(df["quantity"].mode().iloc[0])
df["price"] = df["price"].fillna(df["price"].mean())

df_categ=df.groupby(df["category"])["price"].sum()

from matplotlib import pyplot as plt
plt.bar(df_categ.index.astype(str),df_categ)
plt.show()

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error

x=df[["price"]]
y=df["quantity"]
model=LinearRegression()
kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("Mean Squared Error:",mse)
print("R2 Score:",r2)



