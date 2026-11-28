import pandas as pd
df=pd.read_csv("sales_data.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import mean_squared_error,r2_score
from matplotlib import pyplot as plt

df["Quantity"]=df["Quantity"].fillna(df["Quantity"].mean())
df["Sales"]=df["Sales"].fillna(df["Quantity"]*df["Price"])

dict={'Electoronics':'Electronics',
    'HomeGoods':'Home Goods'}

df["Category"]=df["Category"].replace(dict)

df["Category"]=df["Category"].fillna(df["Category"].mode()[0])

df["Price"]=df["Price"].astype(int)

df["Date"]=pd.to_datetime(df["Date"])

df_monthly=df.groupby(df["Date"].dt.to_period("M"))["Sales"].sum()
df_monthly=df_monthly.sort_values(ascending=False)
print(df_monthly)

df_categ=df.groupby(df["Category"])["Sales"].sum()
df_categ=df_categ.sort_values(ascending=False)
print(df_categ)

print(df_categ.idxmax())
print(df_categ.index[0])


plt.bar(df_categ.index,df_categ.values )
plt.xlabel("Category")
plt.ylabel("Sales")
plt.title("Sales by Category")
plt.show()

model=LinearRegression()
x=df[["Price","Quantity"]]
y=df["Sales"]

kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("Cross-validation scores:",scores)   


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_pred,y_test)
r2=r2_score(y_pred,y_test)
print("Mean Squared Error:",mse)
print("R-squared:",r2)


