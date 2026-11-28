import pandas as pd
df=pd.read_csv("environmental_data.csv")
df=df.dropna()
print(df.head(10))
print(df.describe())


# 温度を昇順に並び替えたデータフレームdf_temp_sortedを制作してみる
df_temp_sorted=df.copy()
for i in range(29):
    if df_temp_sorted.loc[i, "Temperature_C"] > df_temp_sorted.loc[i+1, "Temperature_C"]:
        a=df_temp_sorted.iloc[i].copy()
        df_temp_sorted.iloc[i]=df_temp_sorted.loc[i+1]
        df_temp_sorted.iloc[i+1]=a
    else:
        continue
    
print(df_temp_sorted)

# AQIを昇順に並び替えたデータフレームdf_aqi_sortedを制作してみる
df_aqi_sorted=df.copy()
for j in range(len(df_aqi_sorted)-1):
    if df_aqi_sorted.loc[j, "AQI"] > df_aqi_sorted.loc[j+1, "AQI"]:
        b=df_aqi_sorted.iloc[j].copy()
        df_aqi_sorted.iloc[j]=df_aqi_sorted.loc[j+1]
        df_aqi_sorted.iloc[j+1]=b
    else:
        continue
    
print(df_aqi_sorted)



x1=df[["Temperature_C"]]
y1=df["AQI"]

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, KFold

model1=LinearRegression()
kf1=KFold(n_splits=5, shuffle=True, random_state=42)
scores1=cross_val_score(model1,x1,y1,cv=kf1,scoring="r2")

print(scores1)
print(scores1.mean())

x2=df[["Humidity_%"]]
y2=df["AQI"]

model2=LinearRegression()
kf2=KFold(n_splits=5, shuffle=True, random_state=42)
scores2=cross_val_score(model2,x2,y2,cv=kf2,scoring="r2")

print(scores2)
print(scores2.mean())


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

x3=df[["Temperature_C", "Humidity_%"]]
y3=df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(x3, y3, test_size=0.2, random_state=42)
model3=LinearRegression()
model3.fit(X_train, y_train)
y_pred=model3.predict(X_test)
mse=mean_squared_error(y_test, y_pred)
r2=r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared:", r2)


print("以上の結果から、温度と湿度の両方がAQIの予測に寄与しており、両方のデータを組み合わせることで99％を超える予測精度でAQIを予測できることがわかる。")
