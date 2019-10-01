#DATA PREPROCESSING

#IMPORTING LIBRARIES
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

#IMPORTING THE DATASET
dataset = pd.read_csv('Data.csv')

#CREATING HTE MATRIX OF FEATURES
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
x = pd.DataFrame(x)
y = pd.DataFrame(y)

x.columns = ['Country', 'Age', 'Salary']
y.columns = ['Purchased']



#------------------------------------DEALING WITH MISSING DATA------------------------------------#
from sklearn.preprocessing import Imputer

imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

imputer = imputer.fit(x.values[:, 1:3])
x.values[:, 1:3] = imputer.transform(x.values[:, 1:3])



#------------------------------------ENCODING CATEGORICAL VARIABLES------------------------------------#
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#FOR INDEPENDENT VARIABLE
labelencoder_x = LabelEncoder()
x.values[:, 0] = labelencoder_x.fit_transform(x.values[:, 0])
#CREATING DUMMY VARIABLES FOR CATEGORICAL VARIABLE
onehotencoder = OneHotEncoder(categorical_features = [0])
x = onehotencoder.fit_transform(x).toarray()

#DUMMY VARIABLE TRAP
#x = x[:, 1:]

#FOR DEPENDENT VARIABLE
labelencoder_y = LabelEncoder()
y.values[:, 0] = labelencoder_y.fit_transform(y.values[:, 0])




#------------------------------------SPLITTING INTO TRAIN AND TEST DATASET------------------------------------#
from sklearn.cross_validation import train_test_split

#from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)




#------------------------------------FEATURE SCALING------------------------------------#
# 1. STANDARDIZATION 
# 2. NORMALIZATION

from sklearn.preprocessing import StandardScaler
scl_x = StandardScaler()
x_train = scl_x.fit_transform(x_train)
x_test = scl_x.transform(x_test)
