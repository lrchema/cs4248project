import pandas as pd
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

df = pd.read_csv('../dataset/raw_data/fulltrain.csv', header = None)

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

def main():
    # Test data
    data = {'text': ['Tom to is this the loves this car 59%', '1000000 Joseph is playing amazingly', 'Krish is running great this MORNING!!!', 'John owes me $100']}  
  
    # Create DataFrame  
    data = pd.DataFrame(data)  
    # Test clean
    data['clean'] = preprocess(data['text'])


    # Actual DF
    df['clean'] = preprocess(df[1])

    # df.head()

    # Export to csv
    df.to_csv('../dataset/prep.csv') 


# Lets you run this code
if __name__ == "__main__":
    main()
