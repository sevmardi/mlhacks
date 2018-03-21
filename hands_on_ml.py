import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

oecd_bli = pd.read_csv("data/oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("data/gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding='latin1', na_values="n/a")

#prepare the data 

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

#visualize the data 
country_stats.plot(kind='scatter', x="GDP per capita", y='Life satisfaction')
plt.show()

lin_reg_model = sklearn.linear_model.LinearRegression()
#train the model 
lin_reg_model.fit(X, y)

#make a prediction for cyprus 
x_new = [[22587]] # Cyprus' GDP per capita
print(lin_reg_model.predict(x_new))
