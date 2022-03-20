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

def predict(model, testX):
    pred = model.predict(testX)
    return pred

def read_file(filename):
    return pd.read_csv(filename)

def load_model(filename):
    return pickle.load(open(filename, 'rb'))

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


df = read_file("../dataset/testprep.csv")
# print(df)
length_features(df)
sentiment(df)

model = load_model('trained_model_LR.sav')

vec = TfidfVectorizer()

trainDf = read_file("../dataset/prep.csv")
trainTfidf = vec.fit_transform(trainDf['clean']).toarray()

testTfidf = vec.transform(df['clean']).toarray()
testTfidf = pd.DataFrame(testTfidf)

testFinal = pd.merge(testTfidf,df[features],left_index=True, right_index=True)
y_test = df["0"]

y_pred = predict(model, testFinal)

# evaluate model
score = f1_score(y_test, y_pred, average='macro')
print('score on validation = {}'.format(score))