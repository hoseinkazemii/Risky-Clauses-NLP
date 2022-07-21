import pandas as pd
from sklearn.utils import shuffle

df = pd.read_csv('Dataset.csv')

df_1 = df[df['risk']==1]
df_0 = df[df['risk']==0].loc[:40,:]


df = pd.concat([df_0, df_1], axis=0)
df = shuffle(df)
df.to_csv('sample.csv', encoding = 'utf-8-sig')
