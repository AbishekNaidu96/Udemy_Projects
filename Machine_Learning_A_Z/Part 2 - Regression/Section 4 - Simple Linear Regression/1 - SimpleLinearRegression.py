#SIMPLE LINEAR REGRESSION

#IMPORTING LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset = pd.read_csv('Salary_Data.csv')

#CREATING HTE MATRIX OF FEATURES
#x = pd.DataFrame(dataset.iloc[:, :-1].values)
x = dataset.iloc[:, 0].values
y = dataset.iloc[:, 1].values

#x.columns = ['']
#y.columns = ['']



#------------------------------------SPLITTING INTO TRAIN AND TEST DATASET------------------------------------#
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

x_train = x_train.reshape(20,1)
y_train = y_train.reshape(20,1)
x_test = x_test.reshape(10,1)
y_test = y_test.reshape(10,1)



#------------------------------------FITTING SIMPLE LINEAR REGRESSION TO TRAINING SET------------------------------------#
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)



#------------------------------------PREDICTING THE TEST SET------------------------------------#
y_pred = regressor.predict(x_test)



#------------------------------------VISUALIZATION------------------------------------#
#Training Set
plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regressor.predict(x_train))
plt.title("Salary vs Experience(Train set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

#Test Set
plt.scatter(x_test, y_test, color = "red")
plt.plot(x_train, regressor.predict(x_train))
plt.title("Salary vs Experience(Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

