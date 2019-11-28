import numpy as np
import pandas as pd
import  matplotlib as plt

veri=pd.read_csv("veriler.csv")

Yas = veri.iloc[:,1:4].values

ulke = veri.iloc[:, 0:1].values
#print(ulke)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
ulke[:, 0] = le.fit_transform(ulke[:, 0])
#print(ulke)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categories='auto')
ulke = one.fit_transform(ulke).toarray()
#print(ulke)

c = veri.iloc[:, -1:].values
#print(c)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
c[:, 0] = le.fit_transform(c[:, 0])
#print(ulke)

from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder(categories='auto')
c = one.fit_transform(c).toarray()
#print(c)

sonuc = pd.DataFrame(data = ulke, index = range(22), columns=['fr','tr','us'])

#print(sonuc)

sonuc2=pd.DataFrame(data= Yas, index= range (22), columns=['boy','kilo','yas '])

#print(sonuc2)

sonuc3=pd.DataFrame(data=c[:, :1], index=range(22), columns=['cinsiyet'])
#print(sonuc3)

s=pd.concat([sonuc, sonuc2], axis=1)
#print(s)

s2=pd.concat([s, sonuc3], axis=1)
#print(s2)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size=0.33, random_state=0)

from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

boy = s2.iloc[:, 3:4].values
#print(boy)
sol = s2.iloc[:, :3]
sag = s2.iloc[:, 4:]

veri = pd.concat([sol,sag],axis=1)
x_train, x_test, y_train, y_test = train_test_split(veri, boy, test_size=0.33, random_state=0)

regressor2 = LinearRegression()
regressor2.fit(x_train, y_train)

y_pred = regressor2.predict(x_test)

print(y_pred)