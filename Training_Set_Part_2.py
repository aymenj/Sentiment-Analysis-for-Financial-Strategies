# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
from Utility_Functions import *

# <codecell>

with open("D:299r/anonymizedtwitterdata.txt.txt") as myfile:
    trainset=[myfile.next() for x in xrange(500001)]#I'm not going to use trainset
    testset = [myfile.next() for x in xrange(5001)]

# <codecell>

clf = pickle.load((open('model_bis.m')))
vectorizer = pickle.load((open('vectorizer.v')))

# <codecell>

df_test = to_pd(testset)
df_test = filter_english_tweets(df_test)
test_tweets = list(df_test.tweet)
df_test = clean(df_test)
res = predict(df_test, vectorizer, clf, proba = True)

# <codecell>

print_top10(vectorizer, clf, ['Sad','Happy'])

# <codecell>

res_f = pd.DataFrame({'tweet': list(test_tweets), 'happy':list(res[:,1]), 'sad':list(res[:,0])})
res_f = res_f.sort('happy',ascending = 0)
res_f.head(20)

# <codecell>

#Manual limit for Happy Tweets: iloc = 252
#Manual limit for Sad Tweets: iloc = -270

# <codecell>

#Output Preparation
l = res_f.tweet.count()
res_f['sentiment'] = [1 for _ in range(253)] + [2 for _ in range(253,2607)] + [0 for _ in range(2607,l)]

# <codecell>

output = res_f[['tweet','sentiment']]

# <codecell>

output.to_csv('training_sets/part_2.csv')

# <codecell>


