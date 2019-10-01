#POLINOMIAL LINEAR REGRESSION

dataset <- read.csv('Position_Salaries.csv')
dataset <- dataset[, 2:3]
#Dataset is too small to split into test and train datasets



#------------------------------------FITTING SIMPLE LINEAR REGRESSION TO DATASET------------------------------------#
lin_reg <- lm(Salary ~., data = dataset)
summary(lin_reg)



#------------------------------------FITTING POLYNOMIAL LINEAR REGRESSION TO DATASET------------------------------------#
dataset$Level2 <- dataset$Level^2
dataset$Level3 <- dataset$Level^3
dataset$Level4 <- dataset$Level^4
poly_reg <- lm(Salary ~., data = dataset)
summary(poly_reg)


#------------------------------------VISUALIZATION------------------------------------#
#Simple Linear Regression
library(ggplot2)
ggplot() + geom_point(aes(dataset$Level, dataset$Salary), color = "RED") +
  geom_line(aes(dataset$Level, predict(lin_reg,dataset)), color = "BLUE") +
  ggtitle("Truth Plot (Linear Regression)") +
  xlab("Years of Experience") +
  ylab("Salary")


#Polynomial Linear Regression
library(ggplot2)
ggplot() + geom_point(aes(dataset$Level, dataset$Salary), color = "RED") +
  geom_line(aes(dataset$Level, predict(poly_reg,dataset)), color = "BLUE") +
  ggtitle("Truth Plot (Polynomial Regression)") +
  xlab("Years of Experience") +
  ylab("Salary")



#------------------------------------PREDICTING A RESULT------------------------------------#
#Predicting for simple linear regression
predict(lin_reg, data.frame(Level = 6.5))

#Predicting for Polynomial linear regression
predict(poly_reg, data.frame(Level = 6.5, Level2 = 6.5^2, Level3 = 6.5^3, Level4 = 6.5^4))
