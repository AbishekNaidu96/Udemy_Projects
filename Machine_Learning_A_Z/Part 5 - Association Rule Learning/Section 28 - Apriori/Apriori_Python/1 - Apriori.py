# APRIORI

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
dataset = pd.read_csv("Market_Basket_Optimisation.csv", header = None)
# Empty list
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
#The [] brackets added inside the append() convert the data into a list format 
    # and then append to transactions
#And the data should be in the form of strings for the APRIORI ALGORIGTHM 



#Training Apriori on the dataset
#SEE R FOR A BETTER UNDERSTANDING OF WHY THE FOLLOWING VALUES ARE SET TO WHAT THEY ARE
from apyori import apriori
rules = apriori(transactions,
                min_support = 0.003,
                min_confidence = 0.2,
                min_lift = 3,
                min_lenght = 2)



# VISUALIZATION
# we do not need to 
results = list(rules)
results[0:6]
