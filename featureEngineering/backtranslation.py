#pip install nlpaug

#pip install torch>=1.6.0 transformers>=4.11.3 sentencepiece

import nlpaug.augmenter.word as naw
aug = naw.BackTranslationAug()

def augment_text(extra):
  def inner(text):
    return aug.augment(text, n=extra)
  return inner

# MAKE SURE THE HEADER OF THE LABEL IS NAMED "y" AND text IS NAMED "raw"
def augment_df(df):
  count = df.groupby('y').size().to_dict()
  max_label_no = count[max(count)]

  for i in [1, 2, 3, 4]:
    mask = (df['y'] == i)
    df.loc[mask, 'raw'] = df.loc[mask, 'raw'].apply(augment_text(max_label_no//count[i]))

  return df.explode('raw', ignore_index=True)

import pandas as pd

filename = "subsetofdataset.csv" 
df = pd.read_csv(filename, header=None)
df = df.rename(columns={0:"y", 1:"raw"})

# MAKE SURE THE HEADER OF THE LABEL IS NAMED "y" AND text IS NAMED "raw"
new_df = augment_df(df)

