import pandas as pd 
import numpy as np
import keras
import numpy as np
import tensorflow as tf
from keras import layers

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import RNN
from keras import backend
from sklearn.model_selection import train_test_split

from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D, SpatialDropout1D
from keras.models import Sequential
from keras.preprocessing import text, sequence

import pickle

## Change this value Accordingly. This is suppose to represent the unique number of tokens we have in our corpus
total_vocabulary = 100000
max_sequence_length = 200
class_label = 4
embedding_dim = 100

class MinimalRNNCell(keras.layers.Layer):

    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(MinimalRNNCell, self).__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True

    def call(self, inputs, states):
        prev_output = states[0]
        h = backend.dot(inputs, self.kernel)
        output = h + backend.dot(prev_output, self.recurrent_kernel)
        return output, [output]

def preprocess(df, text_column_name):
    """
    Creates the 2d vector for each of our sentences
    :param feature_column: basically the column that containts our text sentence
    :param num_words: limit number of words for the tokenizer to the most frequent x amount
    :param max_len: the max length of what we want each sentence to have
    """
    tokenizer = Tokenizer(num_words=total_vocabulary, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df[text_column_name].values)

    pickleFile = 'tokenizerRNN.sav'
    pickle.dump(tokenizer, open(pickleFile, 'wb'))

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df[text_column_name].values)
    X = pad_sequences(X, maxlen=max_sequence_length)
    print('Shape of data tensor:', X.shape)
    return X

def RNN_model(X_train):
    """
    Creating our RNN model. This is just an example skeleton
    :param embedding_output: dimension output of the embedding layer
    :param class_label: number of classes we have
    """
    model = Sequential()
    model.add(Embedding(total_vocabulary, embedding_dim, input_length=X_train.shape[1]))
    # model.add(layers.GRU(256, return_sequences=True))
    # model.add(layers.SimpleRNN(128))
    # # model.add(GlobalMaxPool1D())
    # model.add(Dropout(0.5))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(50, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(class_label, activation='softmax'))

    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(13, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.compile(loss='categorical_crossentropy', 
            optimizer='adam', 
            metrics=['accuracy'])
    
    return model
    # model = keras.Sequential()
    # model.add(layers.Embedding(input_dim=1000, output_dim=64))
    # model.add(layers.GRU(256, return_sequences=True))
    # model.add(layers.SimpleRNN(128))

    # # cell = MinimalRNNCell(32)
    # # layer = RNN(cell)
    # # model.add(layer)

    # model.add(layers.Dense(10))
    # model.compile(loss='categorical_crossentropy', 
    #         optimizer='adam', 
    #         metrics=['accuracy'])
    
    # return model

def train_model(model, X_train, y_train):
    epochs = 1
    batch_size = 64
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1)

def one_hot_vec(y):
    y-=1
    print("y",y)
    one_hot_vector = np.zeros((y.size, y.max()+1))
    one_hot_vector[np.arange(y.size), y] = 1
    print(one_hot_vector.shape)
    return one_hot_vector

def main():
    df = pd.read_csv('../dataset/prep.csv')
    X = preprocess(df, "clean")
    X_train, X_test, y_train, y_test = train_test_split(X, one_hot_vec(df["y"]), test_size = 0.10, random_state = 42)

    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)

    model = RNN_model(X_train)

    pickleFile = 'trainedRNN.sav'
    pickle.dump(model, open(pickleFile, 'wb'))

    train_model(model, X_train, y_train)

    accr = model.evaluate(X_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

main()
