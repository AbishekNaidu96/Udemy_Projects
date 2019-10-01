# ARTIFICIAL NEURAL NETWORK

#------------------------------------------------------------------------------------#
#---------------------------- PART I: DATA PREPROCESSING ----------------------------#
#------------------------------------------------------------------------------------#
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values
X



#ENCODING CATEGORICAL VARIABLES
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#Creating fummy variables
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#Dummy variable trap avoid
X = X[:, 1:]



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling - A MUST WHEN WORKING WITH NEURAL NETWORKS
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#-------------------------------------------------------------------------------------#
#----------------------------- PART II: CREATING THE ANN -----------------------------#
#-------------------------------------------------------------------------------------#

#Import libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initializing the ANN
classifier = Sequential()

#Adding the input and first hidden layers
classifier.add(Dense(output_dim = 6,
                     init = 'uniform',
                     activation = 'relu',
                     input_dim = 11))
# The ADD is used to ADD the different LAYERS in our Neural Network
# The DENSE FUNCTION will INITIALIZE all the weights to values CLOSE TO 0
        # It takes all the information on the STEPS SLIDE as parameters
# OUTPUT_DIM = no. of neurons you want to add in the hidden layers
        # Simple way of doing it - averaeg of the input and output layers
        # Artistic way of doing it - PARAMETER TUNING (Will be taught later)
# INIT - related to STEP 1; Where the initialization is done using a uniform distribution
# ACTIVATION - The ACTIVATION FUNCTION we want to use (RECTIFIER HERE)
# INPUT_DIM - A MUST PARAMETER, it has to be done because the ANN has only now
        # being initialized; 
        # once it is done for the first hidden layer, all the others will take the same value


# Adding the second hidden layer
classifier.add(Dense(output_dim = 6,
                     init = 'uniform',
                     activation = 'relu'))

# Adding the OUTPUT LAYER
classifier.add(Dense(output_dim = 1,
                     init = 'uniform',
                     activation = 'sigmoid'))
# There is only 1 output_dim for the binary caetgorical variable, and we use the 
        # SIGMOID function for the outptu
# If the output variable was a categorical variable with more than 2 values then, 
        # OUTPUT_DIM = 2 (for 3 categorical variables, it is OneHotEncoded)
        # ACTIVATION = 'softmax' (categorical variable with more than 2 categories)


# Compiling the ANN
classifier.compile(optimizer = 'adam',
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])
# ADAM: an algorithm to make our weights, in the NN, the best
# LOSS: the kind of COST FUNCTION, ex. sum of square errors
        # For BINARY OUTCOME: binary_crossentropy
        # More than 2 outcomes: categorical_crossentropy 
# METRICS: the criterion to evaluate the model 
        # (A LIST, however we have only 1 value so we put it in [])


#Fitting the ANN to the training set
classifier.fit(X_train, y_train, 
               batch_size = 10,
               nb_epoch = 100)
# BATCH_SIZE: The sizes of the batches activated before weights are recomputed
# NP_EPOCH: number of epochs to be iterated through
      


  
#------------------------------------------------------------------------------------#
#------------------------ PART II: PREDICTING AND EVALUATING ------------------------#
#------------------------------------------------------------------------------------#

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

# Accuracy 
accuracy = (cm[0,0] + cm[1,1]) / [len(dataset) * 0.2]           # TEST_SIZE = 0.2
accuracy
