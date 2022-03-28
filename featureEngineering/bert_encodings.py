
import pandas as pd

filename = "fulltrain.csv" 
df = pd.read_csv(filename, header=None)
df = df.rename(columns={0:"y", 1:"raw"})
df2 = df.copy()




# run: pip install sentence_transformers

from sentence_transformers import SentenceTransformer

bert_minilm_model = SentenceTransformer('all-MiniLM-L6-v2')

df['bert_em'] = df['raw'].apply(bert_minilm_model.encode)

df.to_csv("minilm_bert_embed.csv")

bert_distilroberta_model = SentenceTransformer('all-distilroberta-v1')

df2['bert_em'] = df2['raw'].apply(bert_distilroberta_model.encode)

df2.to_csv("distilroberta_bert_embed.csv")
