import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veri = pd.read_csv("satislar.csv")

print(veri)
aylar = veri[["Aylar"]]

print(aylar)

satislar = veri[["Satislar"]]

print(satislar)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size=1/3, random_state=0)
'''
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
y_train = sc.fit_transform(y_train)
y_test = sc.fit_transform(y_test)
'''
from  sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin= lr.predict(x_test)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test, lr.predict(x_test))
plt.show()

plt.title("aylara göre satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")