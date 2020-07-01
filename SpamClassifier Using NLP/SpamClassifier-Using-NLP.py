#importing the Dataset
import pandas as pd

messages = pd.read_csv("smsspamcollection/SMSSpamCollection",
                          sep="\t" ,names = ["label","message"])

#Cleaning the Data

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
lm = WordNetLemmatizer()
sw = set(stopwords.words("english"))
corpus = []
for sentence in range(0,len(messages)):
    spamORnot = re.sub('[^a-zA-Z]'," ",messages["message"][sentence])
    spamORnot = spamORnot.lower()
    
    spamORnot = spamORnot.split()
    spamORnot = [ls.lemmatize(word) for word in spamORnot if word not in sw ]
    spamORnot = " ".join(spamORnot)
    corpus.append(spamORnot)
    
#Creating a Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
bw = CountVectorizer(max_features=4000)
X = bw.fit_transform(corpus).toarray()      #training Data

#Creating a TF-IDf Model
"""
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
X = tf.fit_transform(corpus).toarray()      #training Data
"""

y = pd.get_dummies(messages["label"])
y = y.iloc[:,1].values

#Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2 )

#Training model using Naive Bayes Classifier
# P(A/B) = (P(B/A)*P(A)) / P(B)
from sklearn.naive_bayes import MultinomialNB       #works on many classes like 1 class,2 class and so on
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred = spam_detect_model.predict(X_test)

#Check the Accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)

























