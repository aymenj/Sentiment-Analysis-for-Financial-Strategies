# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

%matplotlib inline
from datetime import datetime
import prettyplotlib as pplt
from sklearn import linear_model
from __future__ import division

from Financial_Functions import *
from Utility_Functions import *

# <codecell>

#Remove
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

# <codecell>

twits, df = to_pd_stocktwits('Stocktwits_data/AAPL.txt')

# <codecell>

df = clean(df)

# <codecell>

clf = pickle.load((open('model.m')))
vectorizer = pickle.load((open('vectorizer.v')))

# <codecell>

#1ere verif
res = predict(df, vectorizer, clf, proba = False)

# <codecell>

print 'Ratio of Positive Tweets = %f'% (sum(res==1)/len(res))
print 'Ratio of Negative Tweets = %f'% (sum(res==0)/len(res))
print 'Ratio of Neutral Tweets = %f'% (sum(res==2)/len(res))

# <codecell>

#2eme verif
res_prob = predict(df, vectorizer, clf, proba = True)

# <codecell>

fig,ax = plt.subplots(1)
pplt.hist(ax,res_prob[:,1],bins=50)
plt.title('Distribution of Positive Probabilities')
plt.show()

# <codecell>

#3eme verif
df_res = pd.DataFrame({'date': df.date,'tweet': twits, 'positive': list(res_prob[:,1]), 'negative': list(res_prob[:,0]), 'neutral': list(res_prob[:,2])})
df_res = df_res.sort('positive',ascending=False)

# <codecell>

s = df_res.groupby('date').positive.mean()
fig, ax = plt.subplots(1)
pplt.plot(ax,x= s.index,y=s,linewidth=2)
plt.title('Sentiment of AAPL Stock - 01-Jan to 07-Nov 2013')
plt.show()

# <codecell>

#PRICE DOWNLOAD FUNCTIONS
def getNumpyHistoricalTimeseries(symbol,fromDate, toDate):
    link = 'http://ichart.yahoo.com/table.csv?a='+str(fromDate.month-1)+'&c='+str(fromDate.year)+'&b='+str(fromDate.day)+'&e='+str(toDate.day)+'&d='+str(toDate.month-1)+'&g=d&f='+str(toDate.year)+'&s='+symbol+'&ignore=.csv'
    f = urllib2.urlopen(link)
    header = f.readline().strip().split(",")
    return np.loadtxt(f, dtype=np.float, delimiter=",", converters={0: pl.datestr2num})
        

def close_prices(symbol,fromDate,toDate):
    link = 'http://ichart.yahoo.com/table.csv?a='+str(fromDate.month-1)+'&c='+str(fromDate.year)+'&b='+str(fromDate.day)+'&e='+str(toDate.day)+'&d='+str(toDate.month-1)+'&g=d&f='+str(toDate.year)+'&s='+symbol+'&ignore=.csv'
    f = urllib2.urlopen(link)
    f.readline()
    prices = []
    dates = []
    for line in f:
        prices.append(line.strip().split(',')[4])
        dates.append(datetime.strptime(line.strip().split(',')[0],'%Y-%m-%d'))
        
    return pd.Series(prices, index=dates)


from datetime import date
#Get the Close Prices from Yahoo! Finance
start_date = datetime(2013,1,1)
end_date = datetime(2013,11,7)
price = close_prices('AAPL',start_date,end_date)

# <codecell>

#Create the array of Sentiment and Prices
price_sentiment = pd.concat([price.astype(float),s.astype(float)],axis=1).dropna()
price_sentiment.columns = ['price','sentiment']

# <codecell>

fig, ax = plt.subplots(1)
ts_price = price_sentiment.price
pplt.plot(ax,ts_price.index,ts_price,linewidth=2,alpha=0.7,color=(164/256,12/256,12/256))
plt.title('AAPL Close Price Chart - 01 Jan 2013 to 07 Nov 2013')
plt.show()

# <codecell>

def buy_and_hold(df,fromDate,toDate):
    temp = df[fromDate:toDate]
    return (temp.price[1:]-temp.price[0])/temp.price[0]

def maximal_return(df,fromDate,toDate):
    temp = df[fromDate:toDate]
    res = pd.Series([math.fabs(x-y)/y for x,y in zip(temp.price[1:],temp.price[:-1])],index = temp.index[1:]).cumsum()
    return res

def previous_day_sentiment(df,fromDate,toDate,day_lag=1):
    temp = df[fromDate:toDate]
    threshold = temp.sentiment.mean()
    binary = temp.sentiment>=threshold
    ret = [0.]*day_lag
    for i,ele in enumerate(temp.price):
        if i>=(day_lag+1):
            if binary[i-(day_lag+1)]==1:
                ret.append((temp.price[i]-temp.price[i-1])/temp.price[i-1])
            else:
                ret.append((temp.price[i-1]-temp.price[i])/temp.price[i-1])
    return pd.Series(ret,index=temp.price[1:].index).cumsum()

# <codecell>

ret_buy_and_hold = buy_and_hold(price_sentiment,start_date,end_date)
ret_maximal_return = maximal_return(price_sentiment,start_date,end_date)
ret_previous_day_1 = previous_day_sentiment(price_sentiment,start_date,end_date,day_lag=1)
ret_previous_day_2 = previous_day_sentiment(price_sentiment,start_date,end_date,day_lag=2)
ret_previous_day_3 = previous_day_sentiment(price_sentiment,start_date,end_date,day_lag=3)
ret_previous_day_4 = previous_day_sentiment(price_sentiment,start_date,end_date,day_lag=4)
ret_previous_day_5 = previous_day_sentiment(price_sentiment,start_date,end_date,day_lag=5)
ret_previous_day_6 = previous_day_sentiment(price_sentiment,start_date,end_date,day_lag=6)

# <codecell>

fig, ax = plt.subplots(1)
pplt.plot(ax,x=ret_buy_and_hold.index,y=ret_buy_and_hold,linewidth = 2, alpha= 0.7,color= (164/256,12/256,12/256),label='Buy and Hold')
pplt.plot(ax,x=ret_maximal_return.index,y=ret_maximal_return,linewidth = 2,linestyle='--', alpha= 0.7,color= (33/256,6/256,6/256),label='Maximal Return')
pplt.plot(ax,x=ret_previous_day_1.index,y=ret_previous_day_1,linewidth = 2, alpha= 0.7,label='Day-1 Sentiment')
pplt.plot(ax,x=ret_previous_day_2.index,y=ret_previous_day_2,linewidth = 2, alpha= 0.7,label='Day-2 Sentiment')
pplt.plot(ax,x=ret_previous_day_3.index,y=ret_previous_day_3,linewidth = 2, alpha= 0.7,label='Day-3 Sentiment')
pplt.plot(ax,x=ret_previous_day_3.index,y=ret_previous_day_4,linewidth = 2, alpha= 0.7,label='Day-4 Sentiment')
pplt.plot(ax,x=ret_previous_day_3.index,y=ret_previous_day_5,linewidth = 2, alpha= 0.7,label='Day-5 Sentiment')
pplt.plot(ax,x=ret_previous_day_3.index,y=ret_previous_day_6,linewidth = 2, alpha= 0.7,label='Day-6 Sentiment')
pplt.legend(ax,loc='best')
plt.title('Strategy Returns')
plt.show()

# <codecell>

#Linear Regression on Prices with Sentiment as features
#def linear_regression(df, fromDate, toDate, day_lag=2,train_perc=0.75):
df = price_sentiment.copy()
day_lag = 3
#Construct the dataset
temp = df.copy()
for i in range(1,day_lag+1):
    name = 'daylag'+str(i)
    temp[name] = temp.sentiment.shift(i)
temp = temp[(start_date+timedelta(day_lag+1)):end_date]

#Split Train/Test
y = temp.price
train_col = [col for col in temp.columns if col not in ['price']]
X = temp[train_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

#Fit
regr = linear_model.LinearRegression()
regr.fit(X_train, y_train)

#Predict
pred = regr.predict(X_test)

# <codecell>

fig, ax = plt.subplots(1)
pplt.plot(ax,x=range(len(y_test)),y=y_test,linestyle='--',label='Observed Values')
pplt.plot(ax,x=range(len(y_test)),y=pred,label='Predicted Values')
pplt.legend(ax,loc='best')
plt.xlim((0,len(y_test)-1))
plt.title('Price Prediction using Stock Sentiment')
plt.show()
print 'Mean Square Error = %f'% np.mean((pred-y_test)**2)

# <codecell>

#Can I predict the direction??
yes = 0
tot = 0
for i, ele in enumerate(y_test):
    if i>=1:
        if ele>y_test[i-1] and pred[i]>pred[i-1]:
            yes += 1
            tot += 1
        elif ele<y_test[i-1] and pred[i]<pred[i-1]:
            yes += 1
            tot += 1
        else:
            tot += 1
print 'I predict the direction of the price with an accuracy of %f'% (yes/tot)

# <codecell>


