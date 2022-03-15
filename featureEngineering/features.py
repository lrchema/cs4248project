import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
from scipy import sparse
import gensim
import collections 
from collections import Counter
import seaborn as sns

## for sentiment
from textblob import TextBlob

def length_features(df):
	df['word_count'] = df["clean"].apply(lambda x: len(str(x).split(" ")))
	df['char_count'] = df["clean"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
	df['sentence_count'] = df["clean"].apply(lambda x: len(str(x).split(".")))
	df['avg_word_length'] = df['char_count'] / df['word_count']
	df['avg_sentence_length'] = df['word_count'] / df['sentence_count']


def sentiment(df):
	df["sentiment"] = df['clean'].apply(lambda x: 
                   TextBlob(x).sentiment.polarity)

def plot_length_features(df):
	#replace x with what feature we want to visualise
	x, y = "char_count", 0
	fig, ax = plt.subplots(nrows=1, ncols=2)
	fig.suptitle(x, fontsize=12)
	for i in df[y].unique():
    	sns.distplot(df[df[y]==i][x], hist=True, kde=False, 
                 bins=10, hist_kws={"alpha":0.8}, 
                 axlabel="histogram", ax=ax[0])
    	sns.distplot(df[df[y]==i][x], hist=False, kde=True, 
                 kde_kws={"shade":True}, axlabel="density",   
                 ax=ax[1])
	ax[0].grid(True)
	ax[0].legend(df[y].unique())
	ax[1].grid(True)
	plt.show()

def plot_sentiment(df):
	x, y = "sentiment", 0
	fig, ax = plt.subplots(nrows=1, ncols=2)
	fig.suptitle(x, fontsize=12)
	for i in df[y].unique():
    	sns.distplot(df[df[y]==i][x], hist=True, kde=False, 
                 bins=10, hist_kws={"alpha":0.8}, 
                 axlabel="histogram", ax=ax[0])
    	sns.distplot(df[df[y]==i][x], hist=False, kde=True, 
                 kde_kws={"shade":True}, axlabel="density",   
                 ax=ax[1])
	ax[0].grid(True)
	ax[0].legend(df[y].unique())
	ax[1].grid(True)
	plt.show()

