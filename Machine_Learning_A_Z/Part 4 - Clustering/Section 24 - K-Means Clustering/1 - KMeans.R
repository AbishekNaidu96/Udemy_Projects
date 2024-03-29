# K-MEANS CLUSTERING

# Importing the dataset
dataset <- read.csv('Mall_Customers.csv')
X <- dataset[4:5]


#Using the ELBOW method to find the optimal solution 
set.seed(6)
wcss <- vector()
for(i in 1:10) wcss[i] <- sum(kmeans(X, i)$withinss)
plot(1:10, wcss, type ='b', main = paste('Cluster of clients'), xlab = 'Number of clusters', ylab = 'WCSS')



# Applying K-MEANS 
set.seed(29)
kmeans <- kmeans(X, 5, iter.max = 300, nstart = 10)
y_kmeans <- kmeans$cluster



#Visualization
library(cluster)
clusplot(X, kmeans$cluster, lines = 0, shade = TRUE, color = TRUE, labels = 2, plotchar = FALSE,
         span = TRUE, main = "Clusters of Client", xlab = "Annual Income", ylab = "WCSS")
