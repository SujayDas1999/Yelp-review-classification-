#importing the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#reading the csv file 
yelp_df = pd.read_csv("yelp.csv")

#description of yelp dataset
yelp_df.describe()

#some more information on the yelp dataset
yelp_df.info()   #There is no missing data 

#adding length of words to our dataframe

yelp_df['length'] = yelp_df['text'].apply(len)

#now that length column is added lets visualize our histogram 

yelp_df['length'].plot(bins=100,kind='hist')

#lets find out some more information about our length column
yelp_df['length'].describe() #so the maximum word length is 4997 and the minimum word length is 1
#lets see the max word i.e. 4997
yelp_df[yelp_df['length']==4997]['text'].iloc[0]
#lets see our minimum word i.e. 1
yelp_df[yelp_df['length']==1]['text'].iloc[0]

#lets visualise the count plot to see actually how many numbers of 1,2,3,4 and 5 stars do we have
sns.countplot(y='stars',data=yelp_df) 

#lets plot a facetgrid graph

g = sns.FacetGrid(data=yelp_df , col='stars', col_wrap=3)
g.map(plt.hist,'length',bins=20,color='g')

#dividing the data frames into stars 

df_1 = yelp_df[yelp_df['stars']==1]
df_5 = yelp_df[yelp_df['stars']==5]

#concatinating the datasets 
df_all = pd.concat([df_1,df_5]) 

#lets the precentage of stars in the data set

print("The percentage of 1 star reviews are ",(len(df_1)/len(df_all))*100,"%")
print("The percentage of 5 star reviews are ",(len(df_5)/len(df_all))*100,"%")

#ploting the count plot
sns.countplot(df_all['stars'],label='count')

#importing nltk and punctuation library 
import string 
string.punctuation

import nltk
from nltk.corpus import stopwords
#nltk.download('stopwords')
stopwords.words('english')

#lets clean our data set

def message_cleaning(message):
    text_punc_rem = [char for char in message if char not in string.punctuation]
    text_punc_rem_join = ''.join(text_punc_rem)
    text_punc_rem_join_clean = [word for word in text_punc_rem_join.split() if word.lower() not in stopwords.words('english')]
    return text_punc_rem_join_clean

df_clean = df_all['text'].apply(message_cleaning)


#printing the zeroth text value of the cleaned dataframe
print(df_clean[0])

#printing the zeroth text value of the normal dataframe
print(df_all['text'][0])


#applying countvectorizer

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(analyzer = message_cleaning)

cv_df = cv.fit_transform(df_all['text'])

#printing all the feature names
print(cv.get_feature_names())

print(cv_df.toarray())


#dividing our dataset/dataframe into training and testing dataframes
label= df_all['stars'].values
x=cv_df
y= label


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2)

#training the model using naive bayes

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(xtrain,ytrain)


#predicting the training set,test set , calculating its accuracy , finding its classfication report and ploting a heat map
ypred_train = nb.predict(xtrain)
ypred_test = nb.predict(xtest)
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
cm = confusion_matrix(ytrain,ypred_train)
sns.heatmap(cm, annot=True)

print(classification_report(ytrain,ypred_train))
accuracy_score(ytrain,ypred_train)

cms = confusion_matrix(ytest,ypred_test)
sns.heatmap(cms, annot=True)

print(classification_report(ytest,ypred_test))
accuracy_score(ytest,ypred_test)

# [1] represents that the customer is unsatisfied and [5] represents that the customer is happy 
test = input('enter your review here')
test1 = []
test1.append(test)

test1i = cv.transform(test1)
py = nb.predict(test1i)

if(py == [1]):
    print("The customer is not satisfied")
else : 
    print("The customer is satisfied")


