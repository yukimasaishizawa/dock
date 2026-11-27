import math as mt
df=[1,2,3,4,5]
result=sum(df)
print("dfの合計は",result)

df2=range(1000)
result2=sum(df2)
print("df2の合計は",result2)

df3=range(1,1000)
result3=sum(df3)
print("df3の合計は",result3)

df4=range(1,1000,10)
result4=sum(df4)
print("df4の合計は",result4)

df5=range(1,5)
result5=sum(df5)
print("df5の合計は",result5)

print(list(df5))


a=0
for i in range(5):
    a=a+i
print("aの値は",a)

b=0
i=1
while i<5:
    b=b+i
    i=i+1
print("bの値は",b)


#FizzBuzz (1)
#3の倍数の場合，Fizzを出力する
#5の倍数の場合，Buzzを出力する
#15の倍数の場合，FizzBuzzを出力する
#上記以外は，数字をそのまま出力する

#1～30をループし，それぞれの数字に対して上記の条件を適用し，出力する．

for i in range(1,31):
    if i%15==0:
        print("FizzBuzz")
    elif i%3==0:
        print("Fizz")
    elif i%5==0:
        print("Buzz")
    else:
        print(i)



#偶奇判定
#x=input("整数を入力してください:")
#x=int(x)
#if x%2==0:
#    print("偶数です")
#else:    
#    print("奇数です")



#偶奇判定関数作成
def even_odd(x):
    if x%2==0:
        return "偶数です"
    else:
        return "奇数です"
    


#偶奇判定関数のテスト
k=1
while k<30:
    print(k,even_odd(k))
    k=k+1





def fizzbuzz2():
    for j in range(1,31):
        result2=""
        if j%3==0:
            result2="Fizz"
        if j%5==0:
            result2=result2+"Buzz"
        if result2=="":
            result2=i
        print(result2)

print(fizzbuzz2())




