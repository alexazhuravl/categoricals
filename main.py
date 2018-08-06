"""
1. split data into train and test (60:40), export to *.csv
2. look at kaggle approaches
3. look at towardsmachinelearning for the multilabeled classifier
4. n-crossvalidation?
5. lowercase everything
"""
import pandas as pd
import numpy as np
import csv
df = pd.read_csv('/home/az/dev/atvisor/nlp/classifier/data/cat_2018.csv')


df.columns = df.columns.str.lower()
df['categorical'] = pd.Categorical(df['category title (level 2)'])
df_dummies = pd.get_dummies(df['categorical'], prefix = 'category')
new_df = df[['category id (level 2)', 'sub category title (level 3)']].copy()
res = pd.concat([new_df, df_dummies], axis=1)
res.columns = res.columns.str.lower()


