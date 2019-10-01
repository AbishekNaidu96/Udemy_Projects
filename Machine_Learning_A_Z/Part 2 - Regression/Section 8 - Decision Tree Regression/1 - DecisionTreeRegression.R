#DECISION TREE
#Non-linear Non-continuous model

dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]

# Fitting the Decision Tree Regression Model to the dataset
#install.packages("rpart")
library(rpart)
regressor = rpart(Salary ~ ., data = dataset,
                  control = rpart.control(minsplit = 1))

# Predicting a new result
y_pred = predict(regressor, data.frame(Level = 6.5))

# Visualising the Decision Tree Regression Model results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)), colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression Model)') +
  xlab('Level') +
  ylab('Salary')
#The above links each decision tree point to the other by a horizontal line
#It has only 10 point and links that average point value for every interval
#To abdicate the above issue you can create many points that would better display the averages

# Visualising the Decision Tree Regression Model results (for higher resolution and smoother curve)
x_grid = seq(min(dataset$Level), max(dataset$Level), 0.01)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary), colour = 'red') +
  geom_line(aes(x = x_grid, y = predict(regressor, newdata = data.frame(Level = x_grid))), colour = 'blue') +
  ggtitle('Truth or Bluff (Decision Tree Regression Model)') +
  xlab('Level') +
  ylab('Salary')





