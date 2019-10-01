#LOGISTIC REGRESSION

#IMPORTING LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset = pd.read_csv('Social_Network_Ads.csv')

#CREATING HTE MATRIX OF FEATURES
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values



#------------------------------------SPLITTING INTO TRAIN AND TEST DATASET------------------------------------#
from sklearn.cross_validation import train_test_split

#from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)




#------------------------------------FEATURE SCALING------------------------------------#
from sklearn.preprocessing import StandardScaler
scl_X = StandardScaler()
X_train = scl_X.fit_transform(X_train)
X_test = scl_X.transform(X_test)



#------------------------------------FITTING LOGISTIC REGRESSION TO TRAINING SET------------------------------------#
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)



#------------------------------------PREDICTING THE TEST SET RESULTS------------------------------------#
y_pred = classifier.predict(X_test)



#------------------------------------MAKING THE CONFUSION MATRIX------------------------------------#
from sklearn.metrics import confusion_matrix                #This is a function that is imported
#Functions are different from classes; classes have CAPS
cm = confusion_matrix(y_test, y_pred)



#------------------------------------VISUALIZATION------------------------------------#
#TRAINING SET
#Help to colorize the map
from matplotlib.colors import ListedColormap

#A funciton, where we would have to replace the value only once
X_set, y_set = X_train, y_train

#Create own observation points. These points are created as pixled with a 0.01 increment
#We have " -1 " and " +1 " So that the points in the graph are not squeezed together
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

#Create the plot and colorize it
#We make a CONTOUR between the two prediction regions - the red and green 
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

#
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

#
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)

#    
plt.title('Logistic Regression (Training set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()



#TEST SET
X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
    
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()














