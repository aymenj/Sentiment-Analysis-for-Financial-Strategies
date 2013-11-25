__author__ = 'Aymen Jaffry'

import pandas as pd
import nltk
import numpy as np
import string

stopwords = ['a', 'able', 'about', 'across', 'after', 'all', 'almost', 'also', 'am', 'among', 'an', 'and', 'any', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'can', 'cannot', 'could', 'dear', 'did', 'do', 'does', 'either', 'else', 'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has', 'have', 'he', 'her', 'hers', 'him', 'his', 'how', 'however', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'just', 'least', 'let', 'like', 'likely', 'may', 'me', 'might', 'most', 'must', 'my', 'neither', 'no', 'nor', 'not', 'of', 'off', 'often', 'on', 'only', 'or', 'other', 'our', 'own', 'rather', 'said', 'say', 'says', 'she', 'should', 'since', 'so', 'some', 'than', 'that', 'the', 'their', 'them', 'then', 'there', 'these', 'they', 'this', 'tis', 'to', 'too', 'twas', 'us', 'wants', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with', 'would', 'yet', 'you', 'your']
english_vocab = set(w.lower() for w in nltk.corpus.words.words()) # CREATE ENGLISH DICTIONARY

d_smileys = { ':)': 'HAPPYSMILEY1',
             ':-)': 'HAPPYSMILEY2',
             ':o)': 'HAPPYSMILEY3',
             ':(': 'SADSMILEY1',
             ':-(': 'SADSMILEY2',
             ':o(': 'SADSMILEY3'}

def to_pd(file):

    ids=[]
    date=[]
    tweet = []
    for line in file:
        ids.append(line.split('\t')[0])
        date.append(line.split('\t')[1])
        tweet.append(line.split('\t')[2])

    return pd.DataFrame({'ids':ids[1:],'date':pd.to_datetime(date[1:]),'tweet':tweet[1:]})


def filter_english_tweets(df):

    temp = df
    unusual_scores = []
    for i,tweet in enumerate(df.tweet):
        text = nltk.word_tokenize(filter(lambda x: x not in string.punctuation,tweet))
        text_vocab = set(w.lower() for w in text if w.lower().isalpha())
        unusual = text_vocab.difference(english_vocab)
        if len(text)>0:
            unusual_scores.append(np.float(len(unusual))/np.float(len(text)))
        else:
            unusual_scores.append(1.)
    temp['unusual_scores'] = unusual_scores

    temp = temp[temp['unusual_scores']<0.05]
    temp.index = range(len(temp.index))
    return temp.tweet


def filter_smiley_tweets(tweets):

    pos = []
    neg = []
    for tweet in tweets:
        for smiley, placeholder in d_smileys.iteritems():
            tweet = tweet.replace(smiley, placeholder)
        words = nltk.word_tokenize(filter(lambda x: x not in string.punctuation,tweet))
        if 'HAPPYSMILEY1' in words or 'HAPPYSMILEY2' in words or 'HAPPYSMILEY3' in words:
            pos.append(tweet)
        elif 'SADSMILEY1' in words or 'SADSMILEY2' in words or 'SADSMILEY3' in words:
            neg.append(tweet)

    return pos, neg

def tokens(tweets):

    tokens = []
    for tweet in tweets:
        words = nltk.word_tokenize(filter(lambda x: x not in string.punctuation,tweet))
        words = [w for w in words if w not in d_smileys.values()]
        words = [w.lower() for w in words]
        for word in words:
            tokens.append(word)

    tokens = list(set(tokens))
    tokens = [x for x in tokens if x not in stopwords]

    porter_stemmer = nltk.stem.PorterStemmer()
    for i,word in enumerate(tokens):
        temp = porter_stemmer.stem(word)
        tokens[i] = temp

    tokens = list(set(tokens))
    return tokens

def bag_of_words(token_s,tweets):
    d = {}
    for word in token_s:
        d[word] = []

    for tweet in tweets:
        words = nltk.word_tokenize(filter(lambda x: x not in string.punctuation,tweet))
        words = [w.lower() for w in words]
        for word in token_s:
            if word in words:
                d[word].append(1.)
            else:
                d[word].append(0.)

    return pd.DataFrame(d)
