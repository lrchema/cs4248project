import pandas as pd
import collections
import spacy
from textblob import TextBlob

data3 = pd.read_csv("prep.csv", index_col=0)
data = data3.rename(columns={"0":"y", "1":"raw"})

ner = spacy.load("en_core_web_lg")

data["tags"] = data["raw"].apply(lambda x: [(tag.text, tag.label_) 
                                for tag in ner(x).ents] )

def lst_count(lst):
    counter = collections.Counter()
    for x in lst:
        counter[x] += 1
    counter = collections.OrderedDict( 
                     sorted(counter.items(), 
                     key=lambda x: x[1], reverse=True))
    lst_count = [ {key:value} for key,value in counter.items() ]
    return lst_count

data["tags"] = data["tags"].apply(lambda x: lst_count(x))

def ner_features(lst_tuple_count, tag):
    if len(lst_tuple_count) > 0:
        tag_type = []
        for tuple_count in lst_tuple_count:
            for tuple in tuple_count:
                type, n = tuple[1], tuple_count[tuple]
                tag_type = tag_type + [type]*n
                counter = collections.Counter()
                for x in tag_type:
                    counter[x] += 1
        return counter[tag]
    else:
        return 0

tags_set = []
for lst in data["tags"].tolist():
     for counter in lst:
          for k in counter.keys():
              tags_set.append(k[1])
tags_set = list(set(tags_set))
for feature in tags_set:
     data["tags_"+feature] = data["tags"].apply(lambda x: 
                             ner_features(x, feature))

data2 = data[['y', 'raw', 'clean', 'tags_ORG', 'tags_PERSON',
       'tags_ORDINAL','tags_DATE', 'tags_GPE', 'tags_NORP', 'tags_LOC',
       'tags_TIME',
       'tags_CARDINAL']]

def length_features(df):
	df['word_count'] = df["clean"].apply(lambda x: len(str(x).split(" ")))
	df['char_count'] = df["clean"].apply(lambda x: sum(len(word) for word in str(x).split(" ")))
	df['sentence_count'] = df["clean"].apply(lambda x: len(str(x).split(".")))
	df['avg_word_length'] = df['char_count'] / df['word_count']
	df['avg_sentence_length'] = df['word_count'] / df['sentence_count']
	df['unique_word_count'] = df["clean"].apply(lambda x: len(set(str(x).split(" "))))
	df['unique_total_ratio'] = df['unique_word_count'] / df['word_count']

def sentiment(df):
	df["sentiment"] = df['clean'].apply(lambda x: 
                   TextBlob(x).sentiment.polarity)

length_features(data3)

sentiment(data3)

data4 = pd.concat([data3, data2[['tags_ORG', 'tags_PERSON',
       'tags_ORDINAL','tags_DATE', 'tags_GPE', 'tags_NORP', 'tags_LOC',
       'tags_TIME',
       'tags_CARDINAL']]], axis=1)

data4.to_csv("concat.csv")