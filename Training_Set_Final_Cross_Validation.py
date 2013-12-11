# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
from Utility_Functions import *

# <codecell>

part_1 = pd.read_csv('training_sets/part_1.csv')
part_2 = pd.read_csv('training_sets/part_2.csv')
part_3 = pd.read_csv('training_sets/part_3.csv')

part_1 = part_1[['tweet','sentiment']]
part_2 = part_2[['tweet','sentiment']]
part_3 = part_3[['tweet','sentiment']]

final = part_1.append(part_2, ignore_index=True)
final = final.append(part_3, ignore_index=True)

tweets = list(final.tweet)
df = clean(final)

# <codecell>

acc = []
for i in range(10):
    feat_train, feat_test, sentiment_train, sentiment_test = train_test_split(df.tweet, 
                                                                              df.sentiment, 
                                                                              test_size=0.15, 
                                                                              random_state=random.randint(0,100))
    
    train = pd.DataFrame({'tweet':list(feat_train),'sentiment': list(sentiment_train)})
    test = pd.DataFrame({'tweet':list(feat_test),'sentiment': list(sentiment_test)})
    stopwords = stopwords + ['twitter','im','work','new','tri','wa','day','thi','go','out','up','now']
    vectorizer, clf = fit(train,fit_prior = True,binary=True,stopwords = stopwords)
    res = predict(test, vectorizer, clf, proba = False)
    acc.append(float(sum(test.sentiment==res))/test.sentiment.count())
    
print 'Average Accuracy = %f'% np.mean(acc)

# <codecell>

acc = []
for i in range(10):
    feat_train, feat_test, sentiment_train, sentiment_test = train_test_split(part_1.tweet, 
                                                                              part_1.sentiment, 
                                                                              test_size=0.15, 
                                                                              random_state=random.randint(0,100))
    
    train = pd.DataFrame({'tweet':list(feat_train),'sentiment': list(sentiment_train)})
    test = pd.DataFrame({'tweet':list(feat_test),'sentiment': list(sentiment_test)})
    stopwords = stopwords + ['twitter','im','work','new','tri','wa','day','thi','go','out','up','now']
    vectorizer, clf = fit(train,fit_prior = True,binary=True,stopwords = stopwords)
    res = predict(test, vectorizer, clf, proba = False)
    acc.append(float(sum(test.sentiment==res))/test.sentiment.count())
    
print 'Average Accuracy = %f'% np.mean(acc)

# <codecell>

acc = []
for i in range(10):
    feat_train, feat_test, sentiment_train, sentiment_test = train_test_split(part_2.tweet, 
                                                                              part_2.sentiment, 
                                                                              test_size=0.15, 
                                                                              random_state=random.randint(0,100))
    
    train = pd.DataFrame({'tweet':list(feat_train),'sentiment': list(sentiment_train)})
    test = pd.DataFrame({'tweet':list(feat_test),'sentiment': list(sentiment_test)})
    stopwords = stopwords + ['twitter','im','work','new','tri','wa','day','thi','go','out','up','now']
    vectorizer, clf = fit(train,fit_prior = True,binary=True,stopwords = stopwords)
    res = predict(test, vectorizer, clf, proba = False)
    acc.append(float(sum(test.sentiment==res))/test.sentiment.count())
    
print 'Average Accuracy = %f'% np.mean(acc)

# <codecell>

acc = []
for i in range(10):
    feat_train, feat_test, sentiment_train, sentiment_test = train_test_split(part_3.tweet, 
                                                                              part_3.sentiment, 
                                                                              test_size=0.15, 
                                                                              random_state=random.randint(0,100))
    
    train = pd.DataFrame({'tweet':list(feat_train),'sentiment': list(sentiment_train)})
    test = pd.DataFrame({'tweet':list(feat_test),'sentiment': list(sentiment_test)})
    stopwords = stopwords + ['twitter','im','work','new','tri','wa','day','thi','go','out','up','now']
    vectorizer, clf = fit(train,fit_prior = True,binary=True,stopwords = stopwords)
    res = predict(test, vectorizer, clf, proba = False)
    acc.append(float(sum(test.sentiment==res))/test.sentiment.count())
    
print 'Average Accuracy = %f'% np.mean(acc)

# <codecell>


