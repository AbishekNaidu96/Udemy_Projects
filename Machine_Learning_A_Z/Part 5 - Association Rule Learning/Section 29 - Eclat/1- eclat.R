# ECLAT 

# DATA PREPROCESSING
library(arules)
dataset <- read.transactions("Market_Basket_Optimisation.csv", sep = ",", rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN = 10)



# TRAINING THE DATASET 
rules = eclat(dataset, 
                parameter = list(support = 0.004, minlen = 2))



# VISUALIZATION
#sort the rules by decending support
inspect(sort(rules, by = "support")[1:10])







