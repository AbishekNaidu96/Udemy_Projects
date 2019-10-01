#MULTIPLE LINEAR REGRESSION

#IMPORTING LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset = pd.read_csv('50_Startups.csv')

#CREATING HTE MATRIX OF FEATURES
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values
x = pd.DataFrame(x)
y = pd.DataFrame(y)

x.columns = ['R&D Spend', 'Administration', 'Marketing Spend', 'State']
y.columns = ['Profit']



#------------------------------------ENCODING CATEGORICAL VARIABLES------------------------------------#
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#FOR INDEPENDENT VARIABLE
labelencoder_x = LabelEncoder()
x.values[:, 3] = labelencoder_x.fit_transform(x.values[:, 3])
#CREATING DUMMY VARIABLES FOR CATEGORICAL VARIABLE
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#Avoiding the Dummy Variable Trap
x = x[:, 1:]                #The first column is removed



#------------------------------------SPLITTING INTO TRAIN AND TEST DATASET------------------------------------#
from sklearn.cross_validation import train_test_split

#from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)



#------------------------------------FITTING MULTIPLE LINEAR REGRESSION TO TRAINING SET------------------------------------#
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)



#------------------------------------PREDICTING THE TEST SET------------------------------------#
y_pred = regressor.predict(x_test)




#------------------------------------BUILDING OPTIMAL MODEL USING BACKWARD ELIMINATION------------------------------------#
import statsmodels.formula.api as sm
#The statsmodels does not take into consideration the intercept of a regression like b0
#for that reason we have to include a column in the matric of features with just 1, 
#The newly formed column would represent the x0 as the variable of the intercept in the 
#regression equation
#Iteration #1
x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#Iteration #2
#Remove variable x2 with P value > 0.05
x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#Iteration #3
#Remove variable x1 with P value > 0.05
x_opt = x[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#Iteration #4
#Remove variable x1 with P value > 0.05
x_opt = x[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()

#Iteration #5
#Remove variable x1 with P value > 0.05
x_opt = x[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
regressor_OLS.summary()



#------------------------------------AUTOMATIC BACKWARD ELIMINATION WITH P-VALUES------------------------------------#
import statsmodels.formula.api as sm

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
x_opt = x[:, [0, 1, 2, 3, 4, 5]]
x_Modeled = backwardElimination(x_opt, SL)




#------------------------------------AUTOMATIC BACKWARD ELIMINATION WITH P-VALUES AND ADJUSTED R SQUARE------------------------------------#
import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)



