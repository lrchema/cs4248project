from lib2to3.pgen2 import token
import pandas as pd
import numpy as np
import seaborn as sns

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, SpatialDropout2D, GlobalMaxPool1D, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from sklearn.metrics import f1_score

import pickle
import sys
from tensorflow.keras.optimizers import SGD

id = sys.argv[1]

total_vocabulary = 100000
long_sequence_length = 200
short_sequence_length = 40
seq_per_text = long_sequence_length//short_sequence_length

embedding_dim = 100

tokenizer = Tokenizer(num_words=total_vocabulary, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)

def preprocess(df, text_column_name):
    """
    Creates the 2d vector for each of our sentences
    :param feature_column: basically the column that containts our text sentence
    :param num_words: limit number of words for the tokenizer to the most frequent x amount
    :param max_len: the max length of what we want each sentence to have
    """
    tokenizer.fit_on_texts(df[text_column_name].values)
    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    pickleFile = f'tokenizerLSTM-{id}.sav'
    pickle.dump(tokenizer, open(pickleFile, 'wb'))

    # df[text_column_name] = df[text_column_name].apply(lambda x: tokenizer.texts_to_sequences(x))
    # df[text_column_name] = tokenizer.texts_to_sequences(df[text_column_name].values)
    # X = pad_sequences(df[text_column_name], maxlen=long_sequence_length)
    # print("X", X, type(X))
    # df[text_column_name] = X
    # print("df[text_column_name]", df[text_column_name])

    def text_to_sequence(text):
        # print("seq len: ",len(sequence) == 200)
        sequence = tokenizer.texts_to_sequences([text])[0]
        sequence = pad_sequences([sequence], maxlen=long_sequence_length)[0]
        short_seqs = []
        for seq_no in range(seq_per_text):
            short_seqs.append(sequence[seq_no * short_sequence_length : (seq_no + 1) * short_sequence_length])
        return short_seqs    
    df[text_column_name] = df[text_column_name].apply(text_to_sequence)

    return df.explode(text_column_name, ignore_index=True)


def LSTM_model(X_train):
    model = Sequential()
    model.add(Embedding(total_vocabulary, embedding_dim, input_length=X_train.shape[1]))
    # model.add(SpatialDropout2D(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    # model.add(GlobalMaxPool1D())
    # model.add(Dense(4, activation='softmax'))
    # model.add(Dropout(0.6))
    # model.add(Dense(32, activation='silu'))
    # model.add(Dropout(0.4))
    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    epochs = 200
    batch_size = 64
    print(X_train.shape)
    print(y_train.shape)

     # Get sample weights
    y_vec = un_one_hot_vec(y_train)
    sample_weight = np.ones(shape=(len(y_train),))
    count = {}
    for i in range(1, 5):
        count[i] = sum(y==i for y in y_vec)
    print("Count: ", count)
    max_label_no = max(list(count.values()))

    for i in range(1, 5):
        print(max_label_no/count[i])
        sample_weight[y_vec == i] = max_label_no/count[i]
    print("Sample weights: ", sample_weight)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,validation_split=0.2, sample_weight=sample_weight,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])
    return history

def one_hot_vec(y):
    y-=1
    print("y",y)
    one_hot_vector = np.zeros((y.size, y.max()+1))
    one_hot_vector[np.arange(y.size), y] = 1
    print(one_hot_vector.shape)
    return one_hot_vector

def un_one_hot_vec(oh):
    """
    Converts y values into one-hot vectors.
    e.g. [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]] -> [1,3,2]

    :param oh: all values to be converted
    """
    return np.argmax(oh, axis=1) + 1

def main():
    dataframe = pd.read_csv('../dataset/prep.csv')
    df = preprocess(dataframe, "clean")
    X = np.stack(df["clean"].to_numpy())
    X = np.asarray(X).astype(np.int)
    X = tf.convert_to_tensor(X)
    X = np.array(df["clean"].to_list())
    # X = tf.convert_to_tensor(X, dtype=np.int32)
    print(X)

    X_train, X_test, y_train, y_test = train_test_split(X, one_hot_vec(df["y"]), test_size = 0.10, random_state = 42)
    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)

    model = LSTM_model(X_train)

    history = train_model(model, X_train, y_train, X_test, y_test)

    accr = model.evaluate(X_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

    # Save this model
    pickleFile = f'trainedLSTM-{id}.sav'
    pickle.dump(model, open(pickleFile, 'wb'))

    # Plot the loss function
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"LSTMTraining-{id}.jpg")

    # dftest = pd.read_csv("testprep.csv")
    # Xt = tokenizer.texts_to_sequences(dftest["clean"].values)
    # Xt = pad_sequences(Xt, maxlen=max_sequence_length)

    # yt = dftest["y"]
    # ytp_oh = model.predict(Xt)
    # ytp = np.argmax(ytp_oh, axis=1)+1

    # score = f1_score(yt, ytp, average='macro')
    # print('score on validation = {}'.format(score))

    # cm = confusion_matrix(yt, ytp)
    # cm_df = pd.DataFrame(cm,
    #                      index = ['Satire','Hoax','Propaganda', 'Reliable News'],
    #                      columns = ['Satire','Hoax','Propaganda', 'Reliable News'])
    # fig, axs = plt.subplots(3)
    # axs[0] = sns.heatmap(cm_df, annot=True, fmt=".0f")
    # axs[0].title('Confusion Matrix')
    # axs[0].ylabel('Actal Values')
    # axs[0].xlabel('Predicted Values')
    # axs[0].savefig("LSTMcm.jpg")

    # axs[1].plot(history.history['accuracy'])
    # axs[1].plot(history.history['val_accuracy'])
    # axs[1].title('model accuracy')
    # axs[1].ylabel('accuracy')
    # axs[1].xlabel('epoch')
    # axs[1].legend(['train', 'val'], loc='upper left')
    # axs[1].savefig("LSTM_modelacc.jpg")

    # axs[2].plot(history.history['loss'])
    # axs[2].plot(history.history['val_loss'])
    # axs[2].title('model loss')
    # axs[2].ylabel('loss')
    # axs[2].xlabel('epoch')
    # axs[2].legend(['train', 'val'], loc='upper left')
    # axs[2].savefig("LSTM_modelloss.jpg")

main()

# print(df["y"])
# print(df.head())
