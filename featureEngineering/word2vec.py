import pandas as pd
import gensim


url = "./GoogleNews-vectors-negative300.bin"
glove_url = './glove.6B.100d.txt.word2vec'




def generate_word_embeddings(target_df, column_name, embedding_type = "google_news"):
    """
    column name is just the column with the tokenized sentence. 
    Embedding type is the type of embedding we are choosing here. input = [google-news, glove]
    This function will return you a dataframe with a dimension of n depending on the type of embedding used
    Each row represents 1 instance
    """
    if embedding_type == "google-news":
        embeddings = gensim.models.KeyedVectors.load_word2vec_format(url, binary=True)
    
    elif embedding_type == "glove":
        embeddings = gensim.models.KeyedVectors.load_word2vec_format(glove_url, binary=False)

    doc_vectors = pd.DataFrame()
    for sentence in target_df[column_name]:
        temp_df = pd.DataFrame()
        for word in sentence:
            try:
                word_vec = embeddings[word]
                temp_df = temp_df.append(pd.Series(word_vec), ignore_index = True)
            except:
                pass
        
        
        doc_vector = temp_df.mean()
        doc_vectors = doc_vectors.append(doc_vector, ignore_index = True)
    
    doc_vectors = doc_vectors.fillna(0)

    return doc_vectors