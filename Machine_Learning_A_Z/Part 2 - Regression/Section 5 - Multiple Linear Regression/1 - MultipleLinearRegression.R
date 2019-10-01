#MULTIPLE LINEAR REGRESSION

dataset <- read.csv('50_Startups.csv')
str(dataset)

#------------------------------------ENCODING CATEGORICAL VARIABLES------------------------------------#
dataset$State <- factor(dataset$State,
                          levels = c("New York", "California", "Florida"),
                          labels = c(1, 2, 3))



#------------------------------------SPLITTING INTO TRAIN AND TEST DATASET------------------------------------#
#install.packages('caTools')
library(caTools)

set.seed(123)

split <- sample.split(dataset$Profit, SplitRatio = 0.8)

training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)

names(dataset)


#------------------------------------FITTING SIMPLE LINEAR REGRESSION TO TRAINING SET------------------------------------#
regressor <- lm(Profit ~ R.D.Spend + Administration + Marketing.Spend + State, data = training_set)
#OR
regressor <- lm(Profit ~., data = training_set)
summary(regressor)



#------------------------------------PREDICTING THE TEST SET------------------------------------#
y_pred <- predict(regressor, test_set)
y_pred



#------------------------------------BUILDING OPTIMAL MODEL USING BACKWARD ELIMINATION------------------------------------#
regressor <- lm(Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
                data = dataset)
summary(regressor)

regressor <- lm(Profit ~ R.D.Spend + Administration + Marketing.Spend, data = dataset)
summary(regressor)

regressor <- lm(Profit ~ R.D.Spend + Marketing.Spend, data = dataset)
summary(regressor)

regressor <- lm(Profit ~ R.D.Spend, data = dataset)
summary(regressor)



#------------------------------------AUTOMATIC BACKWARD ELIMINATION WITH P-VALUES------------------------------------#
backwardElimination <- function(x, sl) {
  numVars = length(x)
  for (i in c(1:numVars)){
    regressor = lm(formula = Profit ~ ., data = x)
    maxVar = max(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"])
    if (maxVar > sl){
      j = which(coef(summary(regressor))[c(2:numVars), "Pr(>|t|)"] == maxVar)
      x = x[, -j]
    }
    numVars = numVars - 1
  }
  return(summary(regressor))
}

SL = 0.05
dataset = dataset[, c(1,2,3,4,5)]
backwardElimination(training_set, SL)