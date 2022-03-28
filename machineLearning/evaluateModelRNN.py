import pandas as pd
import numpy as np
import seaborn as sns
import pickle

from scipy import sparse
from collections import Counter
import matplotlib.pyplot as plt
from textblob import TextBlob

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout2D
from keras.callbacks import EarlyStopping

from sklearn.metrics import f1_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

############################
#   Constant declaration   #
############################
total_vocabulary = 100000
max_sequence_length = 200
embedding_dim = 100

########################
#   Helper functions   #
########################
def predict(model, testX):
    pred = model.predict(testX)
    return pred

def read_file(filename):
    return pd.read_csv(filename)

def load_model(filename):
    return pickle.load(open(filename, 'rb'))

def load_tokenizer(filename):
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

def preprocess(df, text_column_name, tokenizer=None):
    """
    Creates the 2d vector for each of our sentences
    :param df: dataframe
    :param text_column_name: the name of column to be converted into a vector
    :model: filename of pretrained model
    """
    tokenizer = load_tokenizer(tokenizer)
    # if tokenizer == None:
    #     vec = TfidfVectorizer()
    #     tfidf = vec.fit_transform(df['clean']).toarray()
    # else:
    #     vec = load_tokenizer(tokenizer)
    #     tfidf = vec.transform(df['clean']).toarray()
    
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df[text_column_name].values)
    X = pad_sequences(X, maxlen=max_sequence_length)
    print('Shape of data tensor:', X.shape)
    return X

def train_model(model, X_train, y_train):
    epochs = 5
    batch_size = 64
    print(X_train.shape)
    print(y_train.shape)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    return history

def one_hot_vec(y):
    y-=1
    print("y",y)
    one_hot_vector = np.zeros((y.size, y.max()+1))
    one_hot_vector[np.arange(y.size), y] = 1
    print(one_hot_vector.shape)
    return one_hot_vector


def main():
    df = read_file("../dataset/testprep.csv")

    model = load_model('trainedRNN.sav')

    y_test = df["y"]
    # print(y_test)

    y_pred_oh = predict(model, preprocess(df, "clean", tokenizer="tokenizerRNN.sav"))

    y_pred = np.argmax(y_pred_oh, axis=1)+1
    # evaluate model
    score = f1_score(y_test, y_pred, average='macro')
    print('score on validation = {}'.format(score))

    # generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index = ['Satire','Hoax','Propaganda', 'Reliable News'], 
                         columns = ['Satire','Hoax','Propaganda', 'Reliable News'])
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_df, annot=True, fmt=".0f")
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig("RNNcm.jpg")

main()
