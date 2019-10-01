#SIMPLE LINEAR REGRESSION

dataset <- read.csv('Salary_Data.csv')

#------------------------------------SPLITTING INTO TRAIN AND TEST DATASET------------------------------------#
#install.packages('caTools')
library(caTools)

set.seed(123)

split <- sample.split(dataset$Salary, SplitRatio = 2/3)
training_set <- subset(dataset, split == TRUE)
test_set <- subset(dataset, split == FALSE)



#------------------------------------FEATURE SCALING------------------------------------#
# trainging_set[, 2:3] <- scale(trainging_set[, 2:3])
# test_set[, 2:3] <- scale(test_set[, 2:3])



#------------------------------------FITTING SIMPLE LINEAR REGRESSION TO TRAINING SET------------------------------------#
regressor <- lm(Salary ~ YearsExperience, data = training_set)
summary(regressor)


#------------------------------------PREDICTING THE TEST SET------------------------------------#
y_pred <- predict(regressor, test_set)
y_pred


#------------------------------------VISUALIZATION------------------------------------#
library(ggplot2)
#Training set
ggplot() + geom_point(aes(x = training_set$YearsExperience, y = training_set$Salary),
                      color = "RED") +
  geom_line(aes(x = training_set$YearsExperience, predict(regressor, training_set)),
            color = "BLUE") +
  ggtitle("Salary vs Experience(Train set)") +
  xlab("Years of Experience") +
  ylab("Salary")

#Test set
ggplot() + geom_point(aes(x = test_set$YearsExperience, y = test_set$Salary),
                      color = "RED") +
  geom_line(aes(x = training_set$YearsExperience, predict(regressor, training_set)),
            color = "BLUE") +
  ggtitle("Salary vs Experience(Test set)") +
  xlab("Years of Experience") +
  ylab("Salary")

