import pandas as pd 
import numpy as np
import keras
import numpy as np
import tensorflow as tf
import pickle

from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import RNN
from keras import backend
from keras.layers import Input, Dense, LSTM, Embedding
from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D, SpatialDropout2D
from keras.models import Sequential
from keras.preprocessing import text, sequence

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

############################
#   Constant declaration   #
############################
total_vocabulary = 100000 # Total number of tokens you want Tokenizer to keep
max_sequence_length = 200
class_label = 4
embedding_dim = 100

############################
# Preprocessing functions  #
############################
def preprocess_tfidf(text_column):
    """
    Creates the tfidf vector for each of our sentences
    :param text_column: text column that needs to be converted
    """

    vectorizer = TfidfVectorizer()
    trainTfidf = vectorizer.fit_transform(text_column).toarray()

    pickleFile = 'tokenizerRNN.sav'
    pickle.dump(vectorizer, open(pickleFile, 'wb'))

    print('Tf-Idf vectors:', trainTfidf)
    return trainTfidf

def preprocess(text_column):
    """
    Creates the 2d vector for each of our sentences
    where each element in the vector represents a word index of the word appearing in the sentence 
    e.g. I have to go to work -> [0, 0, 0, 0, 1, 2, 3, 4, 3, 5]
    The above assumes that the length of the sequence has to be 10 and is padded with zeros to fit this length.

    :param text_column: text column that needs to be converted
    """

    tokenizer = Tokenizer(num_words=total_vocabulary, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(text_column.values)

    pickleFile = 'tokenizerRNN.sav'
    pickle.dump(tokenizer, open(pickleFile, 'wb'))

    sentence_vector = tokenizer.texts_to_sequences(text_column.values)
    sentence_vector = pad_sequences(sentence_vector, maxlen=max_sequence_length)
    print('Sentence vector:', sentence_vector)
    return sentence_vector


############################
#      Building models     #
############################
def RNN_model(X_train):
    """
    Creating our RNN model.
    :param X_train: Input for the model later.
                    Needed now just to know the size of embedding layer
    """
    model = Sequential()
    model.add(Embedding(total_vocabulary, embedding_dim, input_length=X_train.shape[1]))
    model.add(layers.GRU(256, return_sequences=True))
    model.add(layers.SimpleRNN(128))
    # model.add(GlobalMaxPool1D())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_label, activation='softmax'))

    return model

def LSTM_model(X_train):
    """
    Creating our RNN model using LSTM layer.
    :param X_train: Input for the model later.
                    Needed now just to know the size of embedding layer
    """
    model = Sequential()
    model.add(Embedding(total_vocabulary, embedding_dim, input_length=X_train.shape[1]))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

def RNN_model_with_cell():
    """
    Creating our RNN model from cells.
    """
    model = keras.Sequential()
    model.add(layers.Embedding(input_dim=1000, output_dim=64))
    model.add(layers.GRU(256, return_sequences=True))
    model.add(layers.SimpleRNN(128))

    cell = MinimalRNNCell(32)
    layer = RNN(cell)
    model.add(layer)
    model.add(layers.Dense(10))

    model.compile(loss='categorical_crossentropy', 
            optimizer='adam', 
            metrics=['accuracy'])
    
    return model

############################
#      Model training      #
############################
def train_model(model, X_train, y_train):
    """
    Trains the model using the data

    :param model: model to use
    :param X_train: input
    :param y_train: expected result
    """
    epochs = 1
    batch_size = 64
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)


############################
#       Main function      #
############################
def main():
    # Read and process input into vectors
    df = pd.read_csv('../dataset/prep.csv')
    X = preprocess(df["clean"])

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, one_hot_vec(df["y"]), test_size = 0.10, random_state = 42)

    # Prepare and train RNN model
    model = LSTM_model(X_train)
    train_model(model, X_train, y_train)

    # Save this model
    pickleFile = 'trainedRNN.sav'
    pickle.dump(model, open(pickleFile, 'wb'))

    # Evaluate the results
    accr = model.evaluate(X_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

main()

############################
#      Helper functions    #
############################
def one_hot_vec(y):
    """
    Converts y values into one-hot vectors.
    e.g. [1, 3, 2] -> [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]

    :param y: all values to be converted
    """
    y -= 1
    one_hot_vector = np.zeros((y.size, y.max()+1))
    one_hot_vector[np.arange(y.size), y] = 1

    print("Y transformed into one-hot vector: ", one_hot_vector)
    return one_hot_vector


# Unused class for generating model using RNN cells
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
