import pandas as pd 
import numpy as np
import keras
import numpy as np
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
# import nlpaug.augmenter.word as naw
# aug = naw.BackTranslationAug()

from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import RNN
from keras import backend
from keras.layers import Input, Dense, LSTM, Embedding
from keras.callbacks import EarlyStopping
from keras.layers import Dropout, Activation, Bidirectional, GlobalMaxPool1D, SpatialDropout2D
from keras.models import Sequential
from keras.preprocessing import text, sequence

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

############################
#     Preprocessing.py     #
############################
import nltk
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import TreebankWordTokenizer
import numpy as np
import re 

from nltk.corpus import stopwords

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
STOPWORD = stopwords.words('english')

lem = WordNetLemmatizer()
stem = PorterStemmer()

df = pd.read_csv('../dataset/raw_data/balancedtest.csv', header = None)

# For lemmatization 
def pos_to_morphy(pos_tag):
    """Convert POS tag to Morphy tag for Wordnet to recognise"""
    tag_dict = {"JJ": wordnet.ADJ,
                "NN": wordnet.NOUN,
                "VB": wordnet.VERB,
                "RB": wordnet.ADV}

    # Return tag if found, Noun if not found
    return tag_dict.get(pos_tag[:2], wordnet.NOUN)


def preprocess(document):
#     print(document)
    # Lowercasing all string elements
    document = [doc.lower() for doc in document]
    
    # Basic tokenization
    document = [re.sub(r'[\|/|-]', r' ', doc) for doc in document]
    
    #print(document)
    #         Stop word Removal
    filtered_words = []
    for doc in document:
        lst = [word for word in doc.split() if word not in STOPWORD]
        doc_string = ' '.join(lst)
        filtered_words.append(doc_string)
        
    document = filtered_words
    document = pd.Series(document)
    
    
# #     # ONLY ONE OF STEMMING OR LEMMATIZATION
# #     # Lemmatize to tokens after POS tagging -> Take very long?
#     document = [" ".join([lem.lemmatize(word.lower(), pos=pos_to_morphy(tag)) 
#                           for word, tag in pos_tag(TreebankWordTokenizer().tokenize(doc))]) for doc in document] 
    
    document = [" ".join([stem.stem(word.lower()) for word, tag in pos_tag(TreebankWordTokenizer().tokenize(doc))]) 
                for doc in document] 
    
    
    # Handle numbers: i.e. moneymoney used instead of <money> since will tokenize away the pointy brackets
    # Duplication of term will be used an "unique" term
    document = [re.sub(r'\$ +[0-9]+(.[0-9]+)?', 'moneymoney', doc) for doc in document]
    document = [re.sub(r'dollars?', 'moneymoney', doc) for doc in document]

    document = [re.sub(r'[0-9]+(.[0-9]+)? \%', 'percentpercent', doc) for doc in document]
    document = [re.sub(r'(\w)+ (percentage)+', 'percentpercent', doc) for doc in document]
    document = [re.sub(r'(\w)+ (\%|percent)+', 'percentpercent', doc) for doc in document]
    document = [re.sub(r'((hundred thousands?)|hundreds?|thousands?|millions?|billions?|trillions?)',
                            'numbernumber', doc) for doc in document]
    

#     print((document))

    return document

############################
#   Constant declaration   #
############################
total_vocabulary = 100000 # Total number of tokens you want Tokenizer to keep
max_sequence_length = 200
class_label = 4
embedding_dim = 100

############################
#    Vectorize functions   #
############################
def vectorize_tfidf(text_column):
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

def vectorise(text_column):
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

def un_one_hot_vec(oh):
    """
    Converts y values into one-hot vectors.
    e.g. [[0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]] -> [1,3,2]

    :param oh: all values to be converted
    """
    return np.argmax(oh, axis=1) + 1

############################
#        Upsampling        #
############################
def augment_text(extra):
  def inner(text):
    print("Back translation")
    # return aug.augment(text, n=extra)
  return inner

# MAKE SURE THE HEADER OF THE LABEL IS NAMED "y" AND text IS NAMED "raw"
def augment_df(df):
  count = df.groupby('y').size().to_dict()
  print(count)
  max_label_no = count[max(count)]

  for i in [2, 4]:
    mask = (df['y'] == i)
    df.loc[mask, 'raw'] = df.loc[mask, 'raw'].apply(augment_text(max_label_no//count[i]))

  return df.explode('raw', ignore_index=True)
    
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
    es = EarlyStopping(monitor='val_loss')
    epochs = 200
    batch_size = 64

     # Get sample weights
    y_vec = un_one_hot_vec(y_train)
    sample_weight = np.ones(shape=(len(y_train),))
    count = {}
    for i in range(1, 5):
        count[i] = sum(y==i for y in y_vec)
    print("Count: ", count)
    max_label_no = max(count, key = count.get)

    for i in range(1, 5):
        print(max_label_no/count[i])
        sample_weight[y_vec == i] = max_label_no/count[i]
    print("Sample weights: ", sample_weight)

    training_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, callbacks=[es], sample_weight=sample_weight)
    return training_history

############################
#       Main function      #
############################
def main():
    # Read csv
    df = pd.read_csv('../dataset/prep.csv')
    print("Original df: ", df)

    # Augment/upsample data
    # df = augment_df(df)
    # df.save("augment.csv")
    print("Augmented df: ", df)

    # Preprocess
    # df['clean'] = preprocess(df['raw'])
    print("Augmented and cleaned:", df)

    # Process input into vectors
    X = vectorise(df["clean"])

    # Split into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, one_hot_vec(df["y"]), test_size = 0.10, random_state = 42)

    # Prepare and train RNN model
    model = LSTM_model(X_train)
    history = train_model(model, X_train, y_train)

    # Plot the loss function
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig("RNNTraining.jpg")

    # Save this model
    pickleFile = 'trainedRNN.sav'
    pickle.dump(model, open(pickleFile, 'wb'))

    # Evaluate the results
    accr = model.evaluate(X_test, y_test)
    print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))

main()

############################
#      Unused functions    #
############################
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
