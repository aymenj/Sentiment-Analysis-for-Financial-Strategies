__author__ = 'Aymen Jaffry'
import numpy as np
import nltk
import pickle
import sys
from sklearn.naive_bayes import MultinomialNB

from NLP_Functions import *
from Fit_Model import *

testset = open(sys.argv[0])

testset = to_pd(testset)
testset = filter_english_tweets(testset)

#Create the Bag of Words
token_s = pd.read_csv('C:/Users/Aymen/Dropbox/Harvard/299r/tokens.csv')
token_s.columns = ['inutile', 'tokens']
BoW_test = bag_of_words(token_s.tokens, testset)

#Classify using the classifier
clf = pickle.load((open('C:/Users/Aymen/Dropbox/Harvard/299r/model.m')))
Y_class = clf.predict(BoW_test).index
Y_prob = clf.predict_proba(BoW_test)

d_output = pd.DataFrame({'tweet': list(testset), 'sentiment': Y_class, 'prob_neg': Y_prob[:, 0], 'prob_pos': Y_prob[:, 1], 'prob_neu': Y_prob[:, 2]})

d_output.to_csv(sys.argv[1])
