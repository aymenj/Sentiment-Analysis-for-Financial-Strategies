# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
from Utility_Functions import *

# <codecell>

turk_results = open("C:/Users/Aymen/Dropbox/Harvard/299r/turk-results-2.txt")

ids, score_s = [], []
turk_results.readline()
for line in turk_results:
    temp = line.split('\t')
    ids.append(temp[0])
    score_s.append(np.float(temp[2]))

pd_1 = pd.DataFrame({'id':ids,'score':score_s})

turk_s = open("C:/Users/Aymen/Dropbox/Harvard/299r/turk-2.txt")

ids, tweet_s = [], []
turk_s.readline()
for line in turk_s:
    temp = line.split('\t')
    ids.append(temp[2])
    tweet_s.append(temp[3])

pd_2 = pd.DataFrame({'id':ids,'tweet':tweet_s})

# <codecell>

turk_data = pd.merge(pd_1,pd_2,on='id',how='inner')

# <codecell>

#Je translate les scores du mechanical turk pour qu'ils soient entre 0 et 1
M = turk_data.score.max()
m = turk_data.score.min()
turk_data.score = (turk_data.score - m)/(M-m+0.01)

# <codecell>

df = turk_data.sort('score',ascending=0)

# <codecell>

plt.plot(df.score)
plt.title('Mechanical Turk Scores')
plt.xlabel('Tweets')
plt.ylabel('Scores')
plt.show()

# <codecell>

plt.hist(df.score, bins = 50)
plt.title('Mechanical Turk Scores')
plt.xlabel('Tweets')
plt.ylabel('Scores')
plt.ylim((0,50))
plt.show()

# <codecell>

#Positive Limit: 310
#Negative Limit: 650

# <codecell>

sentiment = [1 for _ in range(311)] + [2 for _ in range(311,650)] + [0 for _ in range(650,999)]
output = pd.DataFrame({'tweet':df.tweet, 'sentiment': sentiment})

# <codecell>

output.to_csv('training_sets/part_3.csv')

