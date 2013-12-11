# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division
#from pattern.web import Twitter
import time
from datetime import date
from datetime import datetime
from datetime import timedelta

import requests
import urllib2
import cookielib
import string
import re
import json
import pickle

import pandas as pd
import numpy as np
import scipy.stats as sps
import collections

import nltk
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import wordnet as wn
from guess_language import guessLanguage
#nltk.download('words')

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import linear_model

import prettyplotlib as pplt
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib as mpl

#colorbrewer2 Dark2 qualitative color table
dark2_colors = [(0.10588235294117647,0.6196078431372549, 0.4666666666666667),
                (0.8509803921568627, 0.37254901960784315, 0.00784313725490196),
                (0.4588235294117647, 0.4392156862745098, 0.7019607843137254),
                (0.9058823529411765, 0.1607843137254902, 0.5411764705882353),
                (0.4, 0.6509803921568628, 0.11764705882352941),
                (0.9019607843137255, 0.6705882352941176, 0.00784313725490196),
                (0.6509803921568628, 0.4627450980392157, 0.11372549019607843)]

rcParams['figure.figsize'] = (21, 12)
rcParams['figure.dpi'] = 150
rcParams['axes.color_cycle'] = dark2_colors
rcParams['lines.linewidth'] = 2
rcParams['axes.facecolor'] = 'white'
rcParams['font.size'] = 14
rcParams['patch.edgecolor'] = 'white'
rcParams['patch.facecolor'] = dark2_colors[0]
rcParams['font.family'] = 'StixGeneral'

# <codecell>

english_vocab = set(w.lower() for w in nltk.corpus.words.words()) # CREATE ENGLISH DICTIONARY

#Stopwords
with open('C:/Users/Aymen/Dropbox/Harvard/299r/stopwords.txt', 'r') as f:
    st = [line.strip() for line in f.readlines()]
    
stopwords = st[0].split(',')

# <codecell>

def to_pd_stocktwits(file_path):
    f = open(file_path)
    dates = []
    twits = []
    for i,line in enumerate(f):
        if i%2==0:
            twits.append(line)
        else:
            dates.append(line)
    
    #Change date format
    for i,ele in enumerate(dates):
        dates[i] = datetime.strptime(ele[1:11],'%Y-%m-%d')
        
    #Everything into a Pandas Dataframe
    df = pd.DataFrame({'date':dates,'tweet':twits})
    
    return twits,df

def to_pd(tweetset):
    #TO PANDAS DATAFRAME
    print 'Converting to Data Frame'
    ids=[]
    date=[]
    tweet = []
    for line in tweetset:
        ids.append(line.split('\t')[0])
        date.append(line.split('\t')[1])
        tweet.append(line.split('\t')[2])
            
    df = pd.DataFrame({'ids':ids[1:],'date':pd.to_datetime(date[1:]),'tweet':tweet[1:]})
    return df

def filter_english_tweets(df):
    #REMOVE NON LATIN LANGUAGES
    print 'Removing Non Latin Languages'
    temp = []
    for i,tweet in enumerate(df.tweet):
        try:
            if unicode(tweet,'utf8')==tweet:
                temp.append(True)
            else:
                temp.append(False)
        except ValueError:
            temp.append(False)
    
    #GUESS LANGUAGE
    print 'Guessing Language'
    data = df[temp]
    temp = []
    for x in data['tweet']:
        try:
            temp.append(guessLanguage(x)=='en')
        except Exception:
            temp.append(False)
    data = data[temp]
    data.index = range(data.shape[0])
    
    return data
        
def prop_filter_english_tweets(df):
    
    temp_list = []
    for i,tweet in enumerate(df.tweet):
        try:
            if unicode(tweet,'utf8')==tweet:
                temp_list.append(True)
            else:
                temp_list.append(False)
        except ValueError:
            temp_list.append(False)

    temp = df[temp_list]
    unusual_scores = []
    for i,tweet in enumerate(temp.tweet):
        text = nltk.word_tokenize(filter(lambda x: x not in string.punctuation,tweet))
        text_vocab = set(w.lower() for w in text if w.lower().isalpha())
        unusual = text_vocab.difference(english_vocab)
        if len(text)!=0:
            unusual_scores.append(np.float(len(unusual))/np.float(len(text)))
        else:
            unusual_scores.append(1.)
    temp['unusual_scores'] = unusual_scores

    temp = temp[temp['unusual_scores']<0.05]
    temp.index = range(len(temp.index))
    return temp
        
def smiley_training(df):

    #FILTER SMILEYS TWEETS
    print 'Filter Smiley Tweets'
    smileys = [':)',':-)',':o)',':P',':D',':o',':(',':-(',':o(',':-[',":'(",':o[',':[']
    temp_tweet = []
    temp_sentiment = []
    
    for i,tweet in enumerate(df.tweet):
        for smiley in smileys[:4]:
            if smiley in tweet: 
                temp_tweet.append(tweet)
                temp_sentiment.append(1.)
        for smiley in smileys[4:]:
            if smiley in tweet: 
                temp_tweet.append(tweet)
                temp_sentiment.append(0.)
               
    train = pd.DataFrame({'tweet':temp_tweet,'sentiment':temp_sentiment})
    print 'Total Number of Smiley Tweets = %f'% train.sentiment.count()
    print 'Number of Happy Tweets = %f'% train.sentiment.sum()
    print 'Happy Ratio = %f'% (train.sentiment.sum()/train.sentiment.count())
    return train


def clean(df):
    temp_df = df.copy()
    #REMOVING DIRTY THINGS
    print 'Removing Dirty Stuff...'
    for i,tweet in enumerate(temp_df.tweet):
        #Remove hyperlinks
        temp = re.sub(r'https?:\/\/.*\/[a-zA-Z0-9]*', '', tweet)
        #Remove quotes
        temp = re.sub(r'&quot;|&amp&#39;|&#39;|&amp;', '', temp)
        #Remove citations
        temp = re.sub(r'@[a-zA-Z0-9]*', '', temp)
        #Remove tickers
        temp = re.sub(r'\$[a-zA-Z0-9]*', '', temp)
        #Remove numbers
        temp = re.sub(r'[0-9]*','',temp)
        temp_df.tweet[i] = temp
        
    #REMOVE PUNCTUATION
    print 'Remove Punctuation'
    temp_df.tweet = [filter(lambda x: x not in string.punctuation,tweet) for tweet in temp_df.tweet]
    
    #PORTER STEMMER
    print 'Apply Porter Stemmer'
    porter_stemmer = nltk.stem.PorterStemmer()
    for i,tweet in enumerate(temp_df.tweet):
        words = nltk.word_tokenize(tweet)
        for j,word in enumerate(words):
            words[j] = porter_stemmer.stem(word)
        temp_df.tweet[i] = ' '.join(words)
        
    return temp_df    
        
def fit(df, fit_prior=False, binary=False, stopwords='english'):
    #TOKENIZE
    print 'Start Tokenizing Using Tf-Idf'
    vectorizer = CountVectorizer(min_df=1,stop_words=stopwords,binary = binary)
    X = vectorizer.fit_transform(df.tweet)
    X = X.toarray()  
    
    #FIT THE MODEL
    print 'Fitting the Model!!'
    clf = MultinomialNB(fit_prior=fit_prior)
    clf.fit(X, list(df.sentiment))
    
    return vectorizer,clf
    

    
def predict(df, vectorizer, clf, proba = False):

        #TOKENIZE
        print 'Transform Test Set based on Vectorizer'
        X = vectorizer.transform(df.tweet)
        #X = X.toarray() 
        
        #PREDICT
        print 'Predicting!'
        if proba==False:
            return clf.predict(X)
        else:
            return clf.predict_proba(X)
        
def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.feature_log_prob_[i,:])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))

# <codecell>


