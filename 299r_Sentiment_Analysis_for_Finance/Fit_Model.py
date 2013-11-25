__author__ = 'Aymen Jaffry'

import numpy as np
import nltk
import pickle
import pandas as pd
from sklearn.naive_bayes import MultinomialNB

from NLP_Functions import *

train = pd.read_csv('data/train_tweets.csv')

###Already filtered
token_s = tokens(train.tweet)
BoW = bag_of_words(token_s,train.tweet)

token_s_output = pd.Series(token_s)
token_s_output.to_csv('data/tokens.csv')

#Add the labels
BoW['label'] = train.score

#Fit the classifier
n_col = BoW.shape[1]-1
X = BoW.iloc[:,:n_col]
Y = BoW.label
clf = MultinomialNB()
clf.fit(X, Y)

pickle.dump(clf,open('model.mm','w'))
