# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 22:51:43 2021

@author: suma
"""
# %%

import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from IPython.display import display

import xgboost as xgb
from nltk.stem import PorterStemmer
import re
import string

# %%

### 1 
df = pd.read_csv('Participants_Data_PLD/train.csv')
# display(df.head())
test_df = pd.read_csv('Participants_Data_PLD/test.csv')
target_col = 'Loan Status'
categorical_features = ["Batch Enrolled","Grade","Sub Grade","Loan Title","Initial List Status","Employment Duration","Verification Status","Payment Plan","Application Type"]
numerical_features = [col for col in df.columns if  col not in categorical_features and col != 'Loan Status']
for feature in categorical_features:
    df[feature] = df[feature].astype("category")
for feature in categorical_features:
    test_df[feature] = test_df[feature].astype("category")
    
for feature in numerical_features:
    df[feature] = df[feature].astype("float")
for feature in numerical_features:
    test_df[feature] = test_df[feature].astype("float")
    
print(df.info())# %%

df.drop_duplicates(inplace=True)

drop_col = ["ID","Payment Plan","Accounts Delinquent"]
df.drop(drop_col,axis= 1,inplace=True)
# %%
def preprocess_loanTitle(df):
    ps = PorterStemmer()
    def stem_words(x):
        return " ".join([ps.stem(word) for word in x.split()])
    df["Loan Title"] = df["Loan Title"].str.lower()
    
    df["Loan Title"] = df["Loan Title"].apply(lambda x:re.sub('[%s]' % re.escape(string.punctuation),'',x))
    df["Loan Title"] = df["Loan Title"].apply(lambda x:stem_words(x))
    df["Loan Title"] = df["Loan Title"].apply(lambda x:re.sub(' +','',x))
    df["Loan Title"] = df["Loan Title"].apply(lambda x: x if x in ["creditcardrefinanc","debtconsolid"] else "other" )
    return df

df = preprocess_loanTitle(df)
print(df.shape)
# %%
def ouliers_processing(df,columns=[]):
    indices = []
    for col in columns:
        Q3 = np.percentile(df[col],75,interpolation = 'midpoint')
        Q1 = np.percentile(df[col],25,interpolation = 'midpoint')
        IQR = Q3 - Q1
        upper = df[col]>= Q3 + 1.5*IQR
        lower = df[col]<= Q1 - 1.5*IQR
        
        indices.extend(list(np.where(upper)[0]))
        indices.extend(list(np.where(lower)[0]))
        print(len(set(indices)))
     
# ouliers_processing(df,columns= ["Open Account","Revolving Balance","Total Accounts"])

# Revolving Balance, Total Accounts,Total Recieved Interest, Total Recieved Late Fee, Recoveries,Colleciton Recovery Fee, Total Collection Amount, Total Current Balance , Total Revolving Credit Facitity,Home Ownership

# %%

from sklearn import preprocessing
 
# label_encoder object knows how to understand word labels.
for feature in categorical_features:
    label_encoder = preprocessing.LabelEncoder()
    df[feature] = label_encoder.fit_transform(df[feature])
    test_df[feature] = label_encoder.transform(test_df[feature])

