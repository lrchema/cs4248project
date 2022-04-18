
import pandas as pd

filename = "../dataset/raw_data/fulltrain.csv" 
df = pd.read_csv(filename, header=None)
df = df.rename(columns={0:"y", 1:"raw"})
df2 = df.copy()

def selectMiddle(text):
    textS = text.split(" ")
    start = " ".join(textS[0:256])
    end = " ".join(textS[len(textS)-512:len(textS)])
    return start + " " + end


# run: pip install sentence_transformers

from sentence_transformers import SentenceTransformer

bert_minilm_model = SentenceTransformer('all-MiniLM-L6-v2')

df['raw'] = df['raw'].apply(selectMiddle)

print("Done", df["raw"])

df['bert_em'] = df['raw'].apply(bert_minilm_model.encode)

print("done")

df.to_pickle("minilm_bert_embed2.pkl")

print("done")
bert_distilroberta_model = SentenceTransformer('all-distilroberta-v1')

print("done")
df2['bert_em'] = df2['raw'].apply(bert_distilroberta_model.encode)

print("done")
df2.to_pickle("distilroberta_bert_embed2.pkl")
print("done")
