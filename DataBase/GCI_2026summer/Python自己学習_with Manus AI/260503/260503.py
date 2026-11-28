import pandas as pd
df=pd.read_csv("sales_data.csv")
print(df.head())
print(df.isnull().sum())
print(df.describe())


from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split,cross_val_score,KFold
from sklearn.metrics import mean_squared_error, r2_score

#plt.scatter(df["temperature"],df["sales"])
#plt.show()

def model_predict(x,y):
    model=LinearRegression()
    kf=KFold(n_splits=5,shuffle=True,random_state=42)
    scores=cross_val_score(model,x,y,cv=kf,scoring="r2")
    print(scores)
    model2=LinearRegression()
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True,random_state=42)
    model2.fit(x_train,y_train)
    y_pred=model2.predict(x_test)
    mse=mean_squared_error(y_test,y_pred)
    r2scores=r2_score(y_test,y_pred)
    print(mse)
    print(r2scores)

    return model2

final_model=model_predict(df[["temperature","is_weekend","is_campaign"]],df[["sales"]])

matrix1=df["temperature"].corr(df["sales"])
matrix2=df["is_weekend"].corr(df["sales"])
matrix3=df["is_campaign"].corr(df["sales"])
print(matrix1)
print(matrix2)
print(matrix3)


model_predict(df[["temperature"]],df[["sales"]])
model_predict(df[["is_weekend"]],df[["sales"]])
model_predict(df[["is_campaign"]],df[["sales"]])



x_new=pd.DataFrame([[25,0,0]],columns=["temperature","is_weekend","is_campaign"])
prediction=final_model.predict(x_new)
print(prediction[0])