# NLP!!!!!!!

# Importing the dataset
dataset_original <- read.delim("Restaurant_Reviews.tsv", quote = "", stringsAsFactors = FALSE)



# CLEANING THE DATASET
#install.packages("tm")
library(tm)
#install.packages("SnowballC")                    #For stopwords
library(SnowballC)
# as.character(corpus[[1]])
corpus <- VCorpus(VectorSource(dataset$Review))
corpus <- tm_map(corpus, content_transformer(tolower))
# as.character(corpus[[841]])
corpus <- tm_map(corpus, removeNumbers)
# as.character(corpus[[841]])
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeWords, stopwords())
# as.character(corpus[[1]])
corpus <- tm_map(corpus, stemDocument)
# as.character(corpus[[1]])
corpus <- tm_map(corpus, stripWhitespace)
# as.character(corpus[[841]])



# CREATING THE BAG OF WORDS MODEL
# Sparce matrix / Sparcity / Tokenization
dtm <- DocumentTermMatrix(corpus)
dtm <- removeSparseTerms(dtm, sparse = 0.999)           #99.9% of the most frequent (99.9% of the most 1s)
dtm
dataset <- as.data.frame(as.matrix(dtm))
dataset$Liked <- dataset_original$Liked



# FITTING USING RANDOM FOREST
# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Fitting Random Forest Classification to the Training set
library(randomForest)
classifier = randomForest(training_set[-692], training_set$Liked,
                          ntree = 10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)
cm

82+77/200
