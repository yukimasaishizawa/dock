import pandas as pd
df = pd.read_csv("prog3-data-jp.csv")
print(df.head())
print(df.isnull().sum())

from matplotlib import pyplot as plt

x=df["日付"]
y_tokyo=df["東京"]
y_nemuro=df["根室"]
y_nagano=df["長野"]
plt.figure(figsize=(20,6))
plt.plot(x,y_tokyo,color="red",marker="o",label="Tokyo")
plt.plot(x,y_nemuro,color="blue",marker="x",label="Nemuro")
plt.plot(x,y_nagano,color="green",marker="v",label="Nagano")
plt.xlabel("Date")
plt.legend()
plt.show()
