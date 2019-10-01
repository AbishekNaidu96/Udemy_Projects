# THOMPSON SAMPLING 

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#import file 
dataset = pd.read_csv("Ads_CTR_Optimisation.csv")


#Implementing Thompson Sampling
import random
N = 10000                                           # Observations
d = 10                                              # Columns
ads_selected = []                                   # All ads selected
number_of_rewards_0 = [0] * d
number_of_rewards_1 = [0] * d
total_reward = 0
for n in range (0, N):
    ad = 0                                          # The Column with the highest upper bound
    max_random = 0                                  # Maximum random draw
    for i in range (0, d):
        random_beta = random.betavariate(number_of_rewards_1[i] + 1, number_of_rewards_0[i] + 1)
        if(random_beta > max_random):
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if (reward == 1):
        number_of_rewards_1[ad] = number_of_rewards_1[ad] + 1
    else:
        number_of_rewards_0[ad] = number_of_rewards_0[ad] + 1
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