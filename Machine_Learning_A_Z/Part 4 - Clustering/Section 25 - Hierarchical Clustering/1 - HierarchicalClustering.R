# HIERARCHICAL CLUSTERING

#importing the dataset 
dataset <- read.csv("Mall_Customers.csv")
X <- dataset[, 4:5]



# Using the dendrogram to finnd the optimal number of clusters
dendrogram = hclust(dist(X, method = "euclidean"), method = "ward.D")
plot(dendrogram, main = "Dendrogram", xlab = "Customers", ylab = "Euclidean Distance")
#optimal number of clusters = 5



# Fitting Hierarchical Clustering
hc = hclust(dist(X, method = "euclidean"), method = "ward.D")
y_hc = cutree(hc, 5)
y_hc
#Shows cluster numbers for each customer, (each value represents a customer)



#Visualization
library(cluster)
clusplot(X, y_hc, lines = 0, shade = TRUE, color = TRUE, labels = 2, plotchar = FALSE,
         span = TRUE, main = "Clusters of Client", xlab = "Annual Income", ylab = "Euclidean Distance")
