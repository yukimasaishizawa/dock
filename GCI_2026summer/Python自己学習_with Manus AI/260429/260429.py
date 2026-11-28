import pandas as pd
df=pd.read_csv("sales_data.csv")
print(df.head(10))
print(df.isnull().sum())
df["Sales"]=df["Sales"].fillna(df["Sales"].median())
df["Price"]=df["Price"].fillna(df["Price"].median())
print(df.isnull().sum())

print(type(df["Date"]))

df["Date"]=pd.to_datetime(df["Date"])

monthly_sales=df.groupby(df["Date"].dt.to_period("M"))["Sales"].sum()
print(monthly_sales)

monthly_sales=monthly_sales.sort_values(ascending=True)
print(monthly_sales)


product_sales=df.groupby(df["Product"])["Sales"].sum()
print(product_sales)

product_sales=product_sales.sort_values(ascending=True)
print(product_sales)


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

model1=LinearRegression()
x=df[["Price"]]
y=df[["Revenue"]]
kf=KFold(n_splits=5, shuffle=True, random_state=42)
scores1=cross_val_score(model1,x,y,cv=kf,scoring="r2")
print(scores1)




from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

model2=LinearRegression()
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model2.fit(X_train,y_train)
y_pred=model2.predict(X_test)
r2=r2_score(y_test,y_pred)
mse=mean_squared_error(y_test,y_pred)
print("R2 Score:",r2)
print("Mean Squared Error:",mse)




from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

model3=LinearRegression()
x2=df[["Price", "Sales"]]
y2=df[["Revenue"]]
kf=KFold(n_splits=5, shuffle=True, random_state=42)
scores2=cross_val_score(model1,x,y,cv=kf,scoring="r2")
print(scores2)




from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

model4=LinearRegression()
X_train2,X_test2,y_train2,y_test2=train_test_split(x2,y2,test_size=0.2,random_state=42)
model2.fit(X_train2,y_train2)
y_pred2=model2.predict(X_test2)
r2_2=r2_score(y_test2,y_pred2)
mse2=mean_squared_error(y_test2,y_pred2)
print("R2 Score:",r2_2)
print("Mean Squared Error:",mse2)