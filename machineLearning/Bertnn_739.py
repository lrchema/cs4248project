import pandas as pd
import numpy as np
import seaborn as sns
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Embedding, SpatialDropout2D, LeakyReLU
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score
# import pickle5 as pickle
from tensorflow.keras.optimizers import Adam
import pickle

total_vocabulary = 100000
max_sequence_length = 200
embedding_dim = 100

unique_id = int(sys.argv[1])

df = pd.read_pickle("distilroberta_bert_embed.pkl")
print(df.head())

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


def Bert_model(X_train):
    model = Sequential()
    model.add(Dense(500))
    model.add(LeakyReLU(alpha=0.1))
#    model.add(Dropout(0.6))
#    model.add(Dense(300))
    model.add(LeakyReLU(alpha=0.1))
#    model.add(Dropout(0.8))
    model.add(Dense(400, activation='silu'))
#    model.add(LeakyReLU(alpha=0.1))
    model.add(Dense(4, activation='softmax'))
    #model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    epochs = 100
    batch_size = 256 
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

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test),callbacks=[EarlyStopping(monitor='val_loss', patience=5, min_delta=0.0001)])
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
    X = df["bert_em"] #preprocess(df, "bert_em")
    #X.apply(tf.convert_to_tensor)
    #tf.convert_to_tensor(X)
    Xnp = np.stack(X.to_numpy())

    print(type(X[0]), type(Xnp))
    print("xnp shape: ", Xnp.shape)

    print(Xnp)
    X_train, X_test, y_train, y_test = train_test_split(Xnp, one_hot_vec(df["y"]), test_size = 0.10, random_state = 42)
    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)

    model = Bert_model(X_train)

    history = train_model(model, X_train, y_train, X_test, y_test)

    pickleFile = f'trainedBert-{unique_id}.sav'
    pickle.dump(model, open(pickleFile, 'wb'))

    accr = model.evaluate(X_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

    plt.figure(1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f"BertTraining-{unique_id}.jpg")

    y_pred_oh = model.predict(X_test)
    y_pred = np.argmax(y_pred_oh, axis=1)+1
    y_test_decoded = np.argmax(y_test, axis=1)+1

    dftest = pd.read_pickle("distilroberta_bert_embed_test.pkl")
    print(dftest)
    print(dftest["bert_em"])
    Xt = np.stack(dftest["bert_em"].to_numpy())
    # Xt = np.asarray(Xt).astype(np.int)
    # print("Xt shape test:", Xt.shape)
    #Xt=tf.convert_to_tensor(Xt)


    yt = dftest["y"]
    ytp_oh = model.predict(Xt)
    print(ytp_oh)
    ytp = np.argmax(ytp_oh, axis=1)+1
    print("yt ", yt)
    print("ytp ", ytp)
    score = f1_score(yt, ytp, average='macro')
    print('score on validation = {}'.format(score))

    cm = confusion_matrix(yt, ytp)
    cm_df = pd.DataFrame(cm,
                         index = ['Satire','Hoax','Propaganda', 'Reliable News'],
                         columns = ['Satire','Hoax','Propaganda', 'Reliable News'])
    plt.figure(2)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm_df, annot=True, fmt=".0f")
    plt.title('Confusion Matrix')
    plt.ylabel('Actal Values')
    plt.xlabel('Predicted Values')
    plt.savefig(f"Bertcm-{unique_id}-{score}.jpg")


main()


