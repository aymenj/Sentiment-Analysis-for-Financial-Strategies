# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
from Utility_Functions import *

# <codecell>

part_1 = pd.read_csv('training_sets/part_1.csv')
part_2 = pd.read_csv('training_sets/part_2.csv')
part_3 = pd.read_csv('training_sets/part_3.csv')

# <codecell>

part_1 = part_1[['tweet','sentiment']]
part_2 = part_2[['tweet','sentiment']]
part_3 = part_3[['tweet','sentiment']]

# <codecell>

final = part_1.append(part_2, ignore_index=True)
final = final.append(part_3, ignore_index=True)

# <codecell>

tweets = list(final.tweet)
df = clean(final)

# <codecell>

stopwords = stopwords + ['twitter','im','work','new','tri','wa','day','thi','go','out','up','now']
vectorizer, clf= fit(df,fit_prior = True,binary=True,stopwords = stopwords)

# <codecell>

print_top10(vectorizer, clf, ['Sad','Neutral','Happy'])

# <codecell>

#Export Model and Vectorizer
#pickle.dump(clf,open('model.m','w'))
#pickle.dump(vectorizer,open('vectorizer.v','w'))

# <codecell>

res = predict(df, vectorizer, clf, proba = False)
print 'The accuracy is %f'% (float(sum(df.sentiment==res))/df.sentiment.count())

# <codecell>

#Just Part_2 Model
tweets = list(part_2.tweet)
df = clean(part_2)
stopwords = stopwords + ['twitter','im','work','new','tri','wa','day','thi','go','out','up','now']
vectorizer, clf= fit(df,fit_prior = True,binary=True,stopwords = stopwords)
pickle.dump(clf,open('model_2.m','w'))
pickle.dump(vectorizer,open('vectorizer_2.v','w'))

# <codecell>

#Part_1 + Part_2 Model
dff = part_1.append(part_2, ignore_index=True)
tweets = list(dff.tweet)
df = clean(dff)
stopwords = stopwords + ['twitter','im','work','new','tri','wa','day','thi','go','out','up','now']
vectorizer, clf= fit(df,fit_prior = True,binary=True,stopwords = stopwords)
pickle.dump(clf,open('model_2_1.m','w'))
pickle.dump(vectorizer,open('vectorizer_2_1.v','w'))

# <codecell>


