import nlpaug.augmenter.word as naw
aug = naw.BackTranslationAug()
import pandas as pd
import random
import sys

classNum = sys.argv[1]
division = sys.argv[2]

limits = [0,0.5,0.75,0.88,0.92,0.96,0.98]

random.seed(10)

def augment_text(text):
    augmented = aug.augment(text, n=2)
    return augmented

def augment_df(df):
    mask = (df['y'] == classNum)

    for i in range(len(mask)):
        if mask[i]:
            randnum = random.random()
            if limits[division]<=randnum<limits[division+1]:
                mask[i] = True
            else:
                mask[i] = False

    df_part = df.loc[mask]
    df_part['raw'] = df_part['raw'].apply(augment_text)

    return df_part.explode('raw', ignore_index=True)

def main():
    df = pd.read_csv('../dataset/raw_data/fulltrain.csv')

    df = augment_df(df)
    df.to_csv(f"job_class{classNum}_{division}.csv", index=False)

main()
