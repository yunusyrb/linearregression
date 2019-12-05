import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veri=pd.read_csv("maaslar.csv")

x=veri.iloc[:,1:2]
y=veri.iloc[:,2:]

from sklearn.linear_model import LinearRegression
lineer=LinearRegression()
lineer.fit(x.values,y.values)

plt.scatter(x.values,y.values,color="green")
plt.plot(x,lineer.predict( x.values),color="blue")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x.values)
print(x_poly)

lig_reg= LinearRegression()
lig_reg.fit(x_poly,y)
plt.scatter(x.values,y.values,color="green")
plt.plot(x.values,lig_reg.predict(poly_reg.fit_transform(x.values)),color="blue")
plt.show()


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x.values)
print(x_poly)

lig_reg= LinearRegression()
lig_reg.fit(x_poly,y)
plt.scatter(x.values,y.values,color="green")
plt.plot(x.values,lig_reg.predict(poly_reg.fit_transform(x.values)),color="blue")
plt.show()