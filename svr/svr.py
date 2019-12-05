import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veri=pd.read_csv("maaslar.csv")
x=veri.iloc[:,1:2]
y=veri.iloc[:,2:]

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
x_olcek=sc1.fit_transform(x.values)
sc2=StandardScaler()
y_olcek=sc2.fit_transform(y.values)

from sklearn.svm import  SVR

svr_reg = SVR(kernel= "rbf")
svr_reg.fit(x_olcek,y_olcek)

plt.scatter(x_olcek,y_olcek,color="green")
plt.plot(x_olcek,svr_reg.predict(x_olcek),color="blue")
plt.show()