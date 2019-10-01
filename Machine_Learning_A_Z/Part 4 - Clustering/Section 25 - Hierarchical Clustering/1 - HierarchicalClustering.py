import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



#Read the csv File
dataset = pd.read_csv("Mall_Customers.csv")
X = dataset.iloc[:, [3,4]].values



#Using the dendogram to find optimal clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
#linkage is the algorithm for the hierarchical clustering
#ward - tries to MINIMIZED THE SUM OF SQUARES ON VARIENCE
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidian Distance")
plt.show()
#optimal number of clusters is 5



#Fitting Hierarchical Clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5,
                             affinity = 'euclidean',   #Distance to do the linkage
                             linkage = 'ward')
y_hc = hc.fit_predict(X)



#Visualizing the Clusters
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = "red", label = "Careful")
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = "blue", label = "Standard")
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = "green", label = "Target")
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = "cyan", label = "Careless")
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = "magenta", label = "Sensible")
plt.title("Clusters of Clients")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1 - 100)")
plt.legend()
plt.show()