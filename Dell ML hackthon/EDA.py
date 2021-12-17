# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 21:00:45 2021

@author: suma
"""
"""

1. Loadind the data into DF
2. Data Cleaning 
    a.1 CAtegorical Univariate Analysis on Target Column
    a.2 Numerical Univariate Analysis on Target Column
    
"""
# %%

import pandas as pd
import  matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from IPython.display import display

import xgboost as xgb
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
# %%

# 2.a.1
y_value_counts = df[target_col].value_counts().values
print(y_value_counts)

plt.pie(y_value_counts,labels=['Non Defaulter','Defaulter'])   

def stack_plot(data, xtick, col2='TARGET', col3='total',ylabel= 'Loans',title = ''):
    ind = np.arange(data.shape[0])
    
    if len(data[xtick].unique())<5:
        plt.figure(figsize=(5,5))
    elif len(data[xtick].unique())>5 & len(data[xtick].unique())<10:
        plt.figure(figsize=(7,7))
    else:
        plt.figure(figsize=(15,15))
    p1 = plt.bar(ind, data[col3].values)
    p2 = plt.bar(ind, data[col2].values)

    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(ticks=ind,rotation=90,labels= list(data[xtick].values))
    plt.legend((p1[0], p2[0]), ('Not Defaulted', 'Defaulted'))
    plt.show()
    


def univariate_barplots(data,col1,col2='Target_col',select_top_n= False,title= 'Chart'):
    ## counts how many people defaulted the loan
    temp = pd.DataFrame(data.groupby(col1)[col2].agg(lambda x: x.eq(1).sum())).reset_index().rename(columns = {col2:'Defaulted count'})  

    

    temp['total'] = data.groupby(col1).size().values
    temp['Defaulted percent'] =  data.groupby(col1)[col2].agg('mean').values
    temp.sort_values(by=['total'],inplace=True, ascending=False)
    
    if select_top_n:
        temp = temp[0:select_top_n]
    
    # stack_plot(temp,xtick=col1,col2='Defaulted count',col3='total',title = title)
    # print(temp.head(5))
    # print('************************')
    return temp

# for col in categorical_features:
    
#     univariate_barplots(df,col,target_col,title=f'{col} Analysis')


# res = univariate_barplots(df,'Batch Enrolled' ,target_col)


# %%
# 2.a.2
# count = 1 
# n = len(numerical_features)
# for col in numerical_features:
#     plt.subplot(n,3,count)
#     sns.scatterplot(x=df.index,y=df[col],hue=df[target_col])
#     count+=1

# plt.show()


# print(df.ID.nunique()) 
# sns.scatterplot(x=df.index,y=df['Loan Amount'],hue=df[target_col])
# sns.scatterplot(x=df.index,y=df['Funded Amount'],hue=df[target_col])
# sns.scatterplot(x=df.index,y=df['Funded Amount Investor'],hue=df[target_col])

# sns.distplot(df[df[target_col] == 0]['Term'])  ## in months
# sns.distplot(df[df[target_col] == 1]['Term'])

# sns.distplot(df[df[target_col] == 0]['Interest Rate'])
# sns.distplot(df[df[target_col] == 1]['Interest Rate'])

# sns.distplot(df[df[target_col] == 0]['Home Ownership'])
# sns.distplot(df[df[target_col] == 1]['Home Ownership'])


# 
# col ='Collection Recovery Fee'
# sns.distplot(df[df[target_col] == 0][col])
# sns.distplot(df[df[target_col] == 1][col])


col ='Total Revolving Credit Limit'
sns.violinplot(x= df[target_col] , y= df[col])
sns.boxplot(x= df[target_col] , y= df[col])
# %%

# %%
for col in df.columns:
    print(df[col].isna().sum()/df.shape[0],col)
# %%
# %%
# %%
# %%
# %%
# %%
# %%

# %%

# %%
y = df["Loan Status"]
X = df.drop(columns="Loan Status")
x_test = test_df.drop(columns="Loan Status")
# %%
num_classes = 2


clf = xgb.XGBClassifier()
     
# X is the dataframe we created in previous snippet
clf.fit(X, y)
# Must use JSON for serialization, otherwise the information is lost
clf.save_model("model1.json")


# %%
# Get a graph

graph = xgb.to_graphviz(clf, num_trees=1)
# Or get a matplotlib axis
ax = xgb.plot_tree(clf, num_trees=1)
# Get feature importances
clf.feature_importances_
# %%
y_test = clf.predict(x_test)


# %%
print(y_test)
y_pred = pd.DataFrame(y_test)
y_pred.rename(columns={0:"Loan Status"})
y_pred.to_csv('sample_submission.csv', index=False)
