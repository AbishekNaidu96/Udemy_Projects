# APRIORI

# DATA PREPROCESSING
dataset <- read.csv("Market_Basket_Optimisation.csv", header = FALSE)

# The "arules" package takes in a SPARSE matrix as its input; 
# a sparse matrix is a large matrix of many 0s, it has very few non 0 values
# SPARCITY - A LARGE NUMBER OF 0s
# There are 120 products, so you would have to create columns for all those 120 products
# Each row would represent each transaction; and every column would represent the product
# If a product was bought during a transaction it would have a value of 1, else 0
# (A CONCEPT SIMILAR TO DUMMY VARIABLES)

# SPARCE MATRIX CREATION
# install.packages("arules")
library(arules)
dataset <- read.transactions("Market_Basket_Optimisation.csv", sep = ",", rm.duplicates = TRUE)

# OUTPUT MESSAGE:
#   distribution of transactions with duplicates:
#   1 
#   5 
# 1 - represents if its duplicate or triplicate, etc XD
# 5 - the number of those duplicates or so

summary(dataset)
itemFrequencyPlot(dataset, topN = 10)



# TRAINING THE DATASET 
rules = apriori(dataset, 
                parameter = list(support = 0.003, confidence = 0.2))
# The support and confidence inputs in the above code are the threshold values
# Support of items i = (number of transactions containting set of items i) / (total number of transactions)
# We consider the minimum products purchased to be 3
# => support = (3 prducts per day * purchased 7 times a week)/(7500 total transactions)
3*7/7500
# We just start with the default value decrease on every iteration (0.8)
# You will see 0 rules created  for 0.8 confidence, so we divide by 2

# Lets consider Support for min. no. of products purchased to be 4
4*7/7500
rules = apriori(dataset, 
                parameter = list(support = 0.004, confidence = 0.2))


# VISUALIZATION
#sort the rules by decending lift
inspect(sort(rules, by = "lift")[1:10])
# If we see rule #6 we can see that "CHOCOLATE" is purchased with "GROUND BEEF", which makes no sense
# that is because "CHOCOLATE" have a high SUPPORT:
# itemFrequencyPlot(dataset, topN = 10)






