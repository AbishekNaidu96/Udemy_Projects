#DATA PREPROCESSING

getwd()
setwd("/Users/arajnarain/Desktop/Udemy/Machine Learning/Machine Learning A-Z Template Folder/Part 1 - Data Preprocessing/Section 2 -------------------- Part 1 - Data Preprocessing --------------------/Data_Preprocessing")

dataset <- read.csv('Data.csv')

#------------------------------------DEALING WITH MISSING DATA------------------------------------#
dataset[is.na(dataset$Age),"Age"] <- mean(dataset[,"Age"], na.rm = T)
dataset[is.na(dataset$Salary), "Salary"] <- mean(dataset[,"Salary"], na.rm = T)
#OR
dataset$Age <- ifelse(is.na(dataset$Age),
                      ave(dataset$Age, 
                          FUN =  function(x) mean(x, na.rm = T)),
                      dataset$Age)
dataset$Salary <- ifelse(is.na(dataset$Salary),
                        ave(dataset$Salary, 
                            FUN =  function(x) mean(x, na.rm = T)),
                        dataset$Salary)



#------------------------------------ENCODING CATEGORICAL VARIABLES------------------------------------#
dataset$Country <- factor(dataset$Country,
                          levels = c("France", "Spain", "Germany"),
                          labels = c(1, 2, 3))
dataset$Purchased <- factor(dataset$Purchased,
                            levels = c("No", "Yes"),
                            labels = c(0, 1))
str(dataset)



#------------------------------------SPLITTING INTO TRAIN AND TEST DATASET------------------------------------#
#install.packages('caTools')
library(caTools)

set.seed(123)

split <- sample.split(dataset$Purchased, SplitRatio = 0.8)

training_set <- dataset[split, ]
test_set <- dataset[!split, ]
#OR
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)



#------------------------------------FEATURE SCALING------------------------------------#
training_set[, 2:3] <- scale(training_set[, 2:3])
test_set[, 2:3] <- scale(test_set[, 2:3])
