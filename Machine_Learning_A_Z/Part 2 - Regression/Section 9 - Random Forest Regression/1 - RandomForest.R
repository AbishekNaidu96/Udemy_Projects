#RANDOM FORESTS
#Non-linear Non-continuous model

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting the Random Forest Regression Model to the dataset
#install.packages("randomForest")
library(randomForest)
set.seed(1234)
regressor = randomForest(x = dataset[1],                        #A dataframe for x value
                         y = dataset$Salary,                    #A vector for y value   
                         ntree = 500)

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Random Forest Regression Model results (for higher resolution and smoother curve)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), colour = 'blue') +
  ggtitle('Truth or Bluff (DRandom Forest Regression Model)') +
  xlab('Level') +
  ylab('Salary')





