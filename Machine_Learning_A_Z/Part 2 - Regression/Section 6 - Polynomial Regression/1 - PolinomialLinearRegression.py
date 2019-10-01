#POLINOMIAL LINEAR REGRESSION

#IMPORTING LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset = pd.read_csv('Position_Salaries.csv')

#CREATING HTE MATRIX OF FEATURES
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values



#We wont split the dataset because we do not have enough data to split into train and test

#------------------------------------FITTING SIMPLE LINEAR REGRESSION TO DATASET------------------------------------#
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)



#------------------------------------FITTING POLYNOMIAL LINEAR REGRESSION TO DATASET------------------------------------#
from sklearn.preprocessing import PolynomialFeatures
#poly_reg = PolynomialFeatures(degree = 1)
#poly_reg = PolynomialFeatures(degree = 2)
#poly_reg = PolynomialFeatures(degree = 3)
poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)



#------------------------------------VISUALIZATION------------------------------------#
#Simple Linear Regression
plt.scatter(x, y, color = "red")
plt.plot(x, lin_reg.predict(x))
plt.title("Truth Plot (Simple Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")

#Polynomial linear regression
plt.scatter(x, y, color = "red")
#plt.plot(x, lin_reg_2.predict(x_poly))
#or
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)))
plt.title("Truth Plot (Polynomial Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")

#improving the polynomial regression above but creating increments of 0.1
#for a better curve
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape(len(x_grid), 1)
plt.scatter(x, y, color = "red")
plt.plot(x_grid, lin_reg_2.predict(poly_reg.fit_transform(x_grid)))
plt.title("Truth Plot (Polynomial Regression)")
plt.xlabel("Level")
plt.ylabel("Salary")



#------------------------------------PREDICTING A RESULT------------------------------------#
#Predicting for simple linear regression
lin_reg.predict(6.5)

#Predicting for Polynomial linear regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))



