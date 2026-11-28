import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,cross_val_score,train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("exercise_data_day23.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())

df["usage_minutes"]=df["usage_minutes"].fillna(df["usage_minutes"].mean())
df["support_tickets"]=df["support_tickets"].fillna(df["support_tickets"].mean())
df["churned"]=df["churned"].fillna(df["churned"].mean())
print(df.isnull().sum())


model=LinearRegression()
x=df[["support_tickets","churned"]]
y=df["usage_minutes"]
kf=KFold(n_splits=4,shuffle=True,random_state=42)
scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
print("cross_val_score:",scores)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
sc=StandardScaler()
sc.fit(x_train)
x_test_std=sc.transform(x_test)
x_train_std=sc.transform(x_train)
model.fit(x_train_std,y_train)
y_pred=model.predict(x_test_std)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
print("mean_squared_error:",mse)
print("r2_score:",r2)
print("coef",model.coef_)
