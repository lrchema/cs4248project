import pandas as pd 
import numpy as np

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import text, sequence

## Change this value Accordingly. This is suppose to represent the unique number of tokens we have in our corpus
total_vocabulary = 100000


def keras_tokenizer(feature_column: pd.DataFrame, num_words: int, max_len: int) -> np.ndarray:
    """
    Creates the 2d vector for each of our sentences
    :param feature_column: basically the column that containts our text sentence
    :param num_words: limit number of words for the tokenizer to the most frequent x amount
    :param max_len: the max length of what we want each sentence to have
    """
    tokenizer = text.Tokenizer(num_words=num_words)
    ## update the internal vocabulary of the tokenizer based on a list of text we send to it.
    tokenizer.fit_on_texts(list(feature_column))

    tokenized_texts = tokenizer.texts_to_sequences(feature_column)
    X = sequence.pad_sequences(tokenized_texts, maxlen=max_len)

    return X


def RNN_model(embedding_output: int, class_label: int):
    """
    Creating our RNN model. This is just an example skeleton
    :param embedding_output: dimension output of the embedding layer
    :param class_label: number of classes we have
    """
    model = Sequential()
    model.add(Embedding(total_vocabulary, embedding_output))
    model.add(LSTM(25, return_sequences=True))
    model.add(GlobalMaxPool1D())
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(50, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(class_label, activation='softmax'))


    model.compile(loss='categorical_crossentropy', 
            optimizer='adam', 
            metrics=['accuracy'])
    
    return model

    



