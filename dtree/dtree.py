import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veri=pd.read_csv("maaslar.csv")

x=veri.iloc[:,1:2]
y=veri.iloc[:,2:]

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor(random_state=0)
dt.fit(x.values,y.values)
z = x.values + 0.5
k = x.values - 0.4
plt.scatter(x.values,y.values,color="red")
plt.plot(x,dt.predict(x.values),color="blue")
plt.plot(x,dt.predict(z),color="yellow")
plt.plot(x,dt.predict(k),color="blue")
plt.show()


from  sklearn.metrics import  r2_score
print("Random Forest R2 değeri: ")
print(r2_score(y.values,dt.predict(x.values)))
