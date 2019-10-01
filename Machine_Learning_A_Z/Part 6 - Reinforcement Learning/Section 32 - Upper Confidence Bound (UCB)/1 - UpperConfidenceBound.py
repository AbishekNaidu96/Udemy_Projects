#UPPER CONFIDENCE BOUND


#SEE RANDOM SELECTION VALUES FIRST TO COMPARE

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import file 
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


#Implementing UCB
import math
N = 10000                                           # Observations
d = 10                                              # Columns
ads_selected = []                                   # All ads selected
number_of_selections = [0] * d
sum_of_rewards = [0] * d
total_reward = 0
for n in range (0, N):
    ad = 0                                          # The Column with the highest upper bound
    max_upper_bound = 0                             # Highest upper bound value for each observation
    for i in range (0, d):
        if(number_of_selections[i] > 0):
            average_reward = sum_of_rewards[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if(upper_bound > max_upper_bound):
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] = number_of_selections[ad] + 1
    reward = dataset.values[n, ad]
    sum_of_rewards[ad] = sum_of_rewards[ad] + reward
    total_reward = total_reward + reward

#If you look at the "ADS_SELECTED" as we progress to the end of it, you will see one particular ad 
    #appearning more than the other - #4 OR the AD5 - This is the AD that should be provided to 
    #all the users 
# We can observe this using "NUMBER_OF_SELECTIONS" 
    


# VISUALIZATION
plt.hist(ads_selected)
plt.title("Histogram of AD Selections")
plt.xlabel("AD")
plt.ylabel("Number of times each Ad was selected")
plt.show()