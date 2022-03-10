import gensim
import pandas as pd


from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn import utils
from typing import List


def tokenizer():
    """
    proxy function: To be changed
    """
    pass


def tag_document(input_df: pd.DataFrame, x_column: str, tag_column: str) -> pd.Series:
    """
    Creates a TaggedDocument in each row to pass into gensim doc2vec model.
    :param input_df: The dataframe we want to convert.
    :x_column: our input variable.
    :tag_column: just a column that represents an integer id of each row.
    """
    return input_df.apply(lambda r: TaggedDocument(words=tokenizer(r[x_column]), tags=[r[tag_column]]), axis=1)


def model_train(input_series: pd.Series, dm: int ,vector_size: int, alpha: int, min_count: int, epoch: int) -> gensim.models.doc2vec.Doc2Vec:
    """
    Trains the Doc2Vec model to generate our word embedding vectors.
    :param input_series: The input after creating our tagged documents.
    :param dm: Defines the training algorithm. If dm=1, distributed memory (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW) is employed.
    :param vector_size: size of our embeddings.
    :param alpha: initial learning rate.
    :param min_count: Ignores all words with total frequency lower than this.
    :param epoch: Number of epochs we want to train our model on
    """
    model = Doc2Vec(dm=dm, vector_size=vector_size, min_count= min_count, alpha=alpha)
    model.build_vocab(input_series.values)

    train_documents  = utils.shuffle(input_series)
    model.train(train_documents,total_examples=len(train_documents), epochs=epoch)

    return model


def vector_for_learning(model: gensim.models.doc2vec.Doc2Vec, input_series: pd.Series) -> List[List[int]]:
    """
    Gets the word embeddings for any input sentences we have to use for model training
    :param model: our Doc2Vec model
    :param input_series: our series of TaggedDocuments
    """
    feature_vectors = [model.infer_vector(doc.words) for doc in input_series]    
    return feature_vectors
