import pandas as pd
df=pd.read_csv("coffee_shop_sales.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

from matplotlib import pyplot as plt
plt.scatter(df["temperature"],df["coffee_sales"])
plt.xlabel("気温")
plt.ylabel("コーヒーの売り上げ")
plt.show()

from matplotlib import pyplot as plt
plt.scatter(df["promotion"],df["coffee_sales"])
plt.xlabel("宣伝度合い")
plt.ylabel("コーヒーの売り上げ")
plt.show()

from matplotlib import pyplot as plt
plt.scatter(df["pastry_sales"],df["coffee_sales"])
plt.xlabel("ペストリーの売上数")
plt.ylabel("コーヒーの売り上げ")
plt.show()

print("気温とペストリーの売上数はコーヒーの売り上げに関して相関がありそうである。")

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error, r2_score

model1=LinearRegression()
x1=df[["temperature"]]
y1=df[["coffee_sales"]]
kf1=KFold(n_splits=5, shuffle=True, random_state=42)
scores1=cross_val_score(model1,x1,y1,cv=kf1,scoring="r2")
print(scores1)

model2=LinearRegression()
x2=df[["pastry_sales"]]
y2=df[["coffee_sales"]]
kf2=KFold(n_splits=5,shuffle=True,random_state=42)
scores2=cross_val_score(model2,x2,y2,cv=kf2,scoring="r2")
print(scores2)

model1_2=LinearRegression()
x1_train,x1_test,y1_train,y1_test=train_test_split(x1,y1,test_size=0.2,random_state=42)
model1_2.fit(x1_train,y1_train)
y1_pred=model1_2.predict(x1_test)
mse1=mean_squared_error(y1_test,y1_pred)
r2scores1=r2_score(y1_test,y1_pred)
print(mse1)
print(r2scores1)

model2_2=LinearRegression()
x2_train,x2_test,y2_train,y2_test=train_test_split(x2,y2,test_size=0.2,random_state=42)
model2_2.fit(x2_test,y2_test)
y2_pred=model2_2.predict(x2_test)
mse2=mean_squared_error(y2_test,y2_pred)
r2scores2=r2_score(y2_test,y2_pred)
print(mse2)
print(r2scores2)






