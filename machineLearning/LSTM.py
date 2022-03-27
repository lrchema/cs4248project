import pandas as pd
import numpy as np
import seaborn as sns

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout2D
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

total_vocabulary = 100000
max_sequence_length = 200
embedding_dim = 100

df = pd.read_csv('../dataset/prep.csv')

def preprocess(df, text_column_name):
    """
    Creates the 2d vector for each of our sentences
    :param feature_column: basically the column that containts our text sentence
    :param num_words: limit number of words for the tokenizer to the most frequent x amount
    :param max_len: the max length of what we want each sentence to have
    """
    tokenizer = Tokenizer(num_words=total_vocabulary, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
    tokenizer.fit_on_texts(df[text_column_name].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    X = tokenizer.texts_to_sequences(df[text_column_name].values)
    X = pad_sequences(X, maxlen=max_sequence_length)
    print('Shape of data tensor:', X.shape)
    return X


def LSTM_model(X_train):
    model = Sequential()
    model.add(Embedding(total_vocabulary, embedding_dim, input_length=X_train.shape[1]))
    # model.add(SpatialDropout2D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train):
    epochs = 1
    batch_size = 64
    print(X_train.shape)
    print(y_train.shape)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

def one_hot_vec(y):
    y-=1
    print("y",y)
    one_hot_vector = np.zeros((y.size, y.max()+1))
    one_hot_vector[np.arange(y.size), y] = 1
    print(one_hot_vector.shape)
    return one_hot_vector

def main():
    X = preprocess(df, "clean")

    X_train, X_test, y_train, y_test = train_test_split(X, one_hot_vec(df["n"]), test_size = 0.10, random_state = 42)
    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)

    model = LSTM_model(X_train)

    train_model(model, X_train, y_train)

    accr = model.evaluate(X_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

    y_pred = model.predict(X_test)
    y_pred=np.argmax(y_pred, axis=1)
    y_test=np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(cm,
                         index = ['Satire','Hoax','Propaganda', 'Reliable News'], 
                         columns = ['Satire','Hoax','Propaganda', 'Reliable News'])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_df, annot=True, fmt=".0f")
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig("LSTMcm.jpg")
main()


