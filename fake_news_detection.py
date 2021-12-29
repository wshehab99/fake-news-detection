#!/usr/bin/env python
# coding: utf-8

# In[10]:


import nltk
from nltk.corpus import stopwords
import nltk.data
from nltk.stem import PorterStemmer
from nltk.util import ngrams
import pandas as pd
import string
from string import punctuation
#read data set
fake=pd.read_csv('G:\college\\LEVEL4\\NLP\\SECTIONS\\fakeNews\\Fake.csv')
true=pd.read_csv('G:\college\\LEVEL4\\NLP\\SECTIONS\\fakeNews\\True.csv')


# In[11]:


fake.head()


# In[12]:


true.head()


# In[13]:


true['label'] = 1
fake['label'] = 0


# In[14]:


true.head()


# In[15]:


fake.head()


# In[16]:


dataSet=pd.concat([true,fake])
#if there is null value will replace with space
dataSet=dataSet.fillna(' ')
dataSet.head()


# In[17]:


#Creat the Corpus
dataSet['text'] = dataSet['title'] + " " + dataSet['text']

#delete columns I don't need
del dataSet['title']
del dataSet['subject']
del dataSet['date']


# In[20]:


stopWords=set(stopwords.words('english'))
punc=list(string.punctuation)


# In[21]:


#remove not word
dataSet.text.str.replace('[^\w\s]','')


# In[22]:


#data preprocessing
#remove stopWords
def remove_stopWords(text):
    t=[]
    for word in text.split(): #tokens
        #stripe() =>removes any spaces or specified characters at the start and end of a string ?!
        if word.strip().lower() not in stopWords:
            t.append(word)
    return " ".join(t)
#remove punctuation
def remove_punctuation(text):
    t=[]
    for word in text.split(): #tokens
        #stripe() =>removes any spaces or specified characters at the start and end of a string
        if word.strip().lower() not in punc:
            t.append(word)
            
    return " ".join(t)
#segmentaion
def seg(txt):
    punkt=nltk.data.load(r"./tokenizers/punkt/english.pickle")
    

    return punkt.tokenize(txt)
#steming
def steming(text):
    t=[]
    porter=PorterStemmer()
    for word in text.split():
        if word.strip() not in stopWords and word not in punc:
            t.append(porter.stem(word))
    return " ".join(t)
#limmatiztion
def limma(text):
    t=[]
    wnl=nltk.WordNetLemmatizer()
    for word in text.split():
        if word.strip() not in stopWords and word not in punc:
            t.append(wnl.lemmatize(word))
    return " ".join(t)

def pre(text):
    
    text=remove_stopWords(text)
    text=remove_punctuation(text)
    text=limma(text)
    return text

        
dataSet['text']=dataSet['text'].apply(pre)


# In[23]:


#applay ngram
def n_grams(text,n):
    
    temp=ngrams(text.split(),n)
    return [" ".join(g) for g in temp] 
    
def count(pairs):
    fd=nltk.FreqDist(pairs)
    
    
    return fd
def applay_Ngram_count(text):
    text=n_grams(text,1)#n=3
    text=count(text)
    return text
pairs=dataSet['text'].apply(applay_Ngram_count)
#co=count(pairs),
print(pairs)


# In[24]:


#feature extraction
#TF-IDF

from sklearn.feature_extraction.text import TfidfVectorizer
vec=TfidfVectorizer()
features=vec.fit_transform(dataSet['text'])
pd.DataFrame(features.toarray(),columns=sorted(vec.vocabulary_.keys()))


# In[25]:


#binary encoding
from sklearn.feature_extraction.text import CountVectorizer
binVec=CountVectorizer(binary=True)
bfeatures=binVec.fit_transform(dataSet['text'])
pd.DataFrame(bfeatures.toarray(),columns=sorted(binVec.vocabulary_.keys()))


# In[26]:


#counting
vec=CountVectorizer(binary=False)
fc=vec.fit_transform(dataSet['text'])
pd.DataFrame(fc.toarray(),columns=sorted(vec.vocabulary_.keys()))


# In[27]:


#naive bayes with td_idf
from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(features,dataSet['label'],test_size=0.2)


# In[28]:


print(x_train.shape)
print(x_test.shape)


# In[29]:


from sklearn.naive_bayes import MultinomialNB
model= MultinomialNB()
model.fit(x_train,y_train)


# In[30]:


import numpy as np
predicted=model.predict(x_test)
print(np.mean(predicted==y_test))


# In[31]:


#naive bayes with binary encoding
x_train,x_test,y_train,y_test=train_test_split(bfeatures,dataSet['label'],test_size=0.2)
model= MultinomialNB()
model.fit(x_train,y_train)


# In[32]:


predicted=model.predict(x_test)
print(np.mean(predicted==y_test))


# In[33]:


#naive bayes with counting
x_train,x_test,y_train,y_test=train_test_split(fc,dataSet['label'],test_size=0.2)
model= MultinomialNB()
model.fit(x_train,y_train)


# In[34]:


predicted=model.predict(x_test)
print(np.mean(predicted==y_test))


# In[ ]:




