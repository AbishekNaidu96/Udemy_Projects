# ARTIFICIAL NEURAL NETWORK

#------------------------------------------------------------------------------------#
#---------------------------- PART I: DATA PREPROCESSING ----------------------------#
#------------------------------------------------------------------------------------#
# Importing the dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]


#ENCODING CATEGORICAL VARIABLES
dataset$Geography = as.numeric(factor(dataset$Geography, levels = c('France', 'Spain', 'Germany'),
                                      labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender, levels = c('Female', 'Male'), labels = c(0, 1)))


# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling - A MUST WHEN WORKING WITH NEURAL NETWORKS
training_set[-11] = scale(training_set[-11])
test_set[-11] = scale(test_set[-11])



#-------------------------------------------------------------------------------------#
#----------------------------- PART II: CREATING THE ANN -----------------------------#
#-------------------------------------------------------------------------------------# 
# FITTING ANN TO THE DATASET
# DEEP LEARNING PACKAGES IN R
    # 1. NEURAL NET: build deep learning models that are regressors and not classifiers
    # 2. M NET: build deep learning models but with only 1 hidden layer
    # 3. DEEP NET: build deep learning models with many hidden layers
    # 4. H20: considered to be better than the others, becasue, 
              # a. (most important) It is an open source software platform, that allows to connect 
                    #to an instance of a computer system allowing to run the model efficiently 
              # b. Offers a lot of options to run the package, choose no. of hidden layers, 
                    #choose neurons in the hidden layers
              # c. Contains a PARAMETER TUNING ARGUMENT, allowing you to choose optimal numbers to
                    #build your deep learning models
#install.packages("h2o")
library(h2o)
#We need to initialize the H2O package with 
        #NTHREADS = the number of cores (-1 = all cores in the system)
h2o.init(nthreads = -1)
classifier = h2o.deeplearning(y = 'Exited',
                              training_frame = as.h2o(training_set),
                              activation = 'Rectifier',
                              hidden = c(6,6),
                              epochs = 100,
                              train_samples_per_iteration = -2)
# TRAINING_FRAME: Dataset to be trained on, which should be of type H2O FRAME(not DATA FRAME)                         
# HIDDEN: Gives the number of hidden layers and the number of neurons in the hidden layers
        #c(6,6) is a vector of size 2 - 2 HIDDEN LAYERS, each having 6 neurons in them
# TRAIN_SAMPLES_PER_ITERATIONS: In simple it is the bathc size;
        # 0: 1 epoch
        # -1: all the data
        # -2: atuomatic / AUTO-TUNING



#------------------------------------------------------------------------------------#
#------------------------ PART II: PREDICTING AND EVALUATING ------------------------#
#------------------------------------------------------------------------------------#

# Predicting the Test set results
prob_pred = as.vector(h2o.predict(classifier, newdata = as.h2o(test_set[-11])))
y_pred = as.integer(prob_pred > 0.5)

# Making the Confusion Matrix
cm = table(test_set[, 11], y_pred)
cm

# Accuracy 
accuracy = (cm[1,1] + cm[2,2]) / (nrow(dataset) * 0.2)        # TEST_SIZE = 0.2
accuracy

# DISCONNECT FROM H2O!!!!!!!!!!!!!!!!!!!
h2o.shutdown()


