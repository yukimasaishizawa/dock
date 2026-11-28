import pandas as pd
df=pd.read_csv("cleaned_sales_data.csv")
print(df.head())
df_dropna=df.dropna()
df_desc=df.describe()
total_sales=(df_dropna["Quantity"])*(df_dropna["Unit_Price"])*(1-(df_dropna["Discount_Applied"]))+(df_dropna["Shipping_Cost"])
df_dropna["Total_Sales"]=total_sales



y=df_dropna["Total_Sales"]
x=df_dropna[["Quantity","Unit_Price","Discount_Applied","Shipping_Cost"]]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score

model=LinearRegression()
kf=KFold(n_splits=5,random_state=42,shuffle=True)

scores=cross_val_score(model,x,y,cv=kf,scoring="r2")

print(scores)




from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)

print(mse)
print(r2)



