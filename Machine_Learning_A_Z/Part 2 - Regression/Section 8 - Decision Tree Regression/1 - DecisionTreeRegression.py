#DECISION TREE

#IMPORTING LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# Fitting the Decision Tree Regression Model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X,y)

# Predicting a new result
y_pred = regressor.predict(6.5)

# Visualising the Decision Tree Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
#The above links each decision tree point to the other by a horizontal line
#It has only 10 point and links that average point value for every interval
#To abdicate the above issue you can create many points that would better display the averages

# Visualising the Decision Tree Regression results (for higher resolution and smoother curve)
#X_grid = np.arange(min(X), max(X), 0.1)
#For even better resolution, let the iterations be in 0.01
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Decision Tree Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()