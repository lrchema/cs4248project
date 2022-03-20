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
import pickle

from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB



## for sentiment
from textblob import TextBlob

features = ['sentiment', 'word_count', 'char_count', 'sentence_count', 'avg_word_length', 'avg_sentence_length', 'unique_word_count', 'unique_total_ratio']

def length_features(df):
	df['word_count'] = df["clean"].apply(lambda x: len(str(x).split(" ")))
	df['char_count'] = df["clean"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
	df['sentence_count'] = df["clean"].apply(lambda x: len(str(x).split(".")))
	df['avg_word_length'] = df['char_count'] / df['word_count']
	df['avg_sentence_length'] = df['word_count'] / df['sentence_count']
	df['unique_word_count'] = df["clean"].apply(lambda x: len(set(str(x).split(" "))))
	df['unique_total_ratio'] = df['unique_word_count'] / df['word_count']


def sentiment(df):
	df["sentiment"] = df['clean'].apply(lambda x: 
                   TextBlob(x).sentiment.polarity)

def plot_length_features(df):
	#replace x with what feature we want to visualise
	x, y = "char_count", 0
	fig, ax = plt.subplots(nrows=1, ncols=2)
	fig.suptitle(x, fontsize=12)
	for i in df[y].unique():
		sns.distplot(df[df[y]==i][x], hist=True, kde=False, bins=10, hist_kws={"alpha":0.8}, axlabel="histogram", ax=ax[0])
		sns.distplot(df[df[y]==i][x], hist=False, kde=True, kde_kws={"shade":True}, axlabel="density", ax=ax[1])
	ax[0].grid(True)
	ax[0].legend(df[y].unique())
	ax[1].grid(True)
	plt.show()

def plot_sentiment(df):
	x, y = "sentiment", 0
	fig, ax = plt.subplots(nrows=1, ncols=2)
	fig.suptitle(x, fontsize=12)
	for i in df[y].unique():
		sns.distplot(df[df[y]==i][x], hist=True, kde=False, bins=10, hist_kws={"alpha":0.8}, axlabel="histogram", ax=ax[0])
		sns.distplot(df[df[y]==i][x], hist=False, kde=True, kde_kws={"shade":True}, axlabel="density", ax=ax[1])
	ax[0].grid(True)
	ax[0].legend(df[y].unique())
	ax[1].grid(True)
	plt.show()

def read_file(filename):
	return pd.read_csv(filename)

def train_model(model, trainX, trainY):
	model.fit(trainX, trainY)
	pickleFile = 'trained_model_NB.sav'
	pickle.dump(model, open(pickleFile, 'wb'))

def predict(model, testX):
    pred = model.predict(testX)
    return pred


# read in training data and add features
df = read_file("../dataset/prepSmall.csv")
# print(df)
length_features(df)
sentiment(df)

vec = TfidfVectorizer()
trainTfidf = vec.fit_transform(df['clean']).toarray()
trainTfidf = pd.DataFrame(trainTfidf)

trainFinal = pd.merge(trainTfidf,df[features],left_index=True, right_index=True)
y_train = df["0"]
print(trainFinal)
print(trainTfidf)
# define model
# TODO: switch up the model name here to train
model = MultinomialNB(max_iter=4000, verbose=1)

train_model(model, trainFinal, y_train)

y_pred = predict(model, trainFinal)


# evaluate model
score = f1_score(y_train, y_pred, average='macro')
print('score on validation = {}'.format(score))