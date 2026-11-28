import pandas as pd
df=pd.read_csv("cafe_sales_data.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())


from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import r2_score,mean_squared_error

df_corr_temp=df["sales"].corr(df["temperature"])
print("temperatureとsalesの相関係数は:",df_corr_temp)

df_corr_weekend=df["sales"].corr(df["is_weekend"])
print("is_weekendとsalesの相関係数は:",df_corr_weekend)

df_corr_event=df["sales"].corr(df["event"])
print("eventとsalesの相関係数は:",df_corr_event)

print("以上より、イベントの有無との相関が最も強いことがわかる。また、休日か平日かも関係がありそうである。")


plt.scatter(df["temperature"],df["sales"])
plt.xlabel("temp")
plt.ylabel("sales")
plt.show()

model=LinearRegression()
x=df[["temperature","is_weekend","event"]]
y=df["sales"]
kf=KFold(n_splits=5,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print(scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
mse=mean_squared_error(y_pred,y_test)
r2score=r2_score(y_pred,y_test)
print(mse)
print(r2score)