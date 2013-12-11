# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division
import urllib2
import cookielib
import pickle
import re
import pandas as pd
import time
from datetime import date
from datetime import datetime
from datetime import timedelta

import numpy as np
import matplotlib.pyplot as plt

import prettyplotlib as pplt
from matplotlib import rcParams
import matplotlib.cm as cm
import matplotlib as mpl

#colorbrewer2 Dark2 qualitative color table
dark2_colors = [(0.10588235294117647, 0.6196078431372549, 0.4666666666666667),
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

