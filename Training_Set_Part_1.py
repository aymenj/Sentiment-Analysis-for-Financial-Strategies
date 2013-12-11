# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
from Utility_Functions import *

# <codecell>

with open("D:299r/anonymizedtwitterdata.txt.txt") as myfile:
    trainset=[myfile.next() for x in xrange(500001)]
    testset = [myfile.next() for x in xrange(10001)]

# <codecell>

df = to_pd(trainset)
df = filter_english_tweets(df)
#tweets = list(df.tweet)
df_u = smiley_training(df)
df = clean(df_u)

# <codecell>

#Export Smiley Training Set
df_u.to_csv('training_sets/part_1.csv')

# <codecell>

#Train
stopwords = stopwords + ['twitter','im','work']
vectorizer, clf= fit(df,fit_prior = True,binary=True,stopwords = stopwords)

# <codecell>

#Export Model and Vectorizer
pickle.dump(clf,open('model_bis.m','w'))
pickle.dump(vectorizer,open('vectorizer_bis.v','w'))

