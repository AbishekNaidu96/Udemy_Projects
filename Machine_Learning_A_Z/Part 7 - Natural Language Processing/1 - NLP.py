# NLP!!!!!!!

# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Importing the dataset
dataset = pd.read_csv("Restaurant_Reviews.tsv", delimiter = '\t', quoting = 3)



# CLEANING THE TEXT
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Remove all NON-LETTERS and replace by a ' ' (Space)
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0])
# Convert to lower case and split into a LIST
review = review.lower()
review = review.split()
# review = [word for word in review if word not in set(stopwords.words('english'))]
# We use a "SET" because the format of "LIST" that it is in would be slower to read from

# STEMMING
ps = PorterStemmer()
review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]

# Join the values
review = ' '.join(review)



# LOOPING FOR ALL VALUES IN THE DATASET
corpus = []
# A corpus in ML is a term given to a collection of texts like reviews or longer like,
    # web scrapped, articles, etc.
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
    
   
# CREATING THE BAG OF WORDS MODEL
    #Sparce matrix / Sparcity / Tokenization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
# We can perform most of the cleaning operationg using the various parameters of the
    # CountVectorizer class
# MAX_FEATURES is the (in this case) 1500 most common features 
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values




# USING NAIVE BAYES TO FIT THE MODEL
# We mostly use Naive Bayes, Decision Tree, or Random Forests for NLP 
# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm

accuracy = (cm[0,0] + cm[1,1]) / [len(dataset) * 0.2]           # TEST_SIZE = 0.2
accuracy









