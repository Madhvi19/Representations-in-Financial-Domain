#!/usr/bin/env python
# coding: utf-8

# In[23]:


import json
import numpy as np
import re
import warnings
import pickle
import concurrent
import contractions
import en_core_web_sm
import logging as log
import multiprocessing
import pickle
import spacy
import sys
import os
import re
import threading
import time
from collections import Counter
from nltk.corpus import stopwords
from nltk import PorterStemmer
from spacy import displacy
from Stemmer import Stemmer
from word2number import w2n
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
import re
from word2number import w2n
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn


# In[24]:


def get_info(id, info_name, data_json):
    return data_json[id]['info'][0][info_name]


# In[25]:


def load_data(file_path):

    with open(file_path, 'r', encoding="utf-8") as file:
        ids = []
        aspects = []
        snippets = []
        companies = []
        sentences = []
        sentiment_scores = []

        data_json = json.load(file)

        def getAspects(aspect):
            aspect = aspect.replace('[', '')
            aspect = aspect.replace(']', '')
            aspect = aspect.replace('\'', '')
            return aspect.split('/')

        for id in data_json:
            ids.append(id.lower())
            companies.append(get_info(id, 'target', data_json).lower())
            sentences.append(data_json[id]['sentence'].lower().lower())
            snippets.append(get_info(id, 'snippets', data_json).lower())
            aspects.append(getAspects(get_info(id, 'aspects', data_json).lower()))
            sentiment_scores.append(float(get_info(id, 'sentiment_score', data_json)))

    return sentences, sentiment_scores


# In[38]:


file_path = sys.argv[1]
sentences, scores = load_data(file_path)

file_path = sys.argv[2]
s, sc = load_data(file_path)
sentences +=s
scores += sc


# In[39]:


def __removePunctuations(sequence,ner_tags):
    #########################################################################################
    # This method removes any punctuations and gets only the text from the given sequence.
    #########################################################################################
    try:
        if sequence is not None and sequence.strip() != "":
            if sequence in ner_tags:
                return re.sub('[^A-Za-z0-9%$.]+',' ',sequence)
            else:
                return re.sub('[^A-Za-z0-9$%]+',' ',sequence)
        return sequence # return sequence as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while removing punctuations in the sequence '{0}'. Error is: {1}; {2}".format(sequence, str(exc_type), str(exc_value))
        raise Exception(err)


# In[40]:


def __clean_data(data,ner_tags):
    #########################################################################################
    # This method cleans the data by applying following API's and returns list of preprocessed tokens
    #     data.lower()
    #     __expandContractions(data)
    #     __removeHtmlTags(data)
    #     __replaceurls(data)
    #     __removePunctuations(data)
    #     data.split(" ")
    #########################################################################################
    #data = __expandContractions(data)
    #data = __removeHtmlTags(data) # remove tags
    #data = __replaceurls(data)
    data = data.replace("$","$ ")
    data = data.replace("%"," % ")
    data = data.split(" ")
    clean = []
    for word in data:
        w_clean = __removePunctuations(word,ner_tags)
        clean.extend(w_clean.split(" "))
    data = clean
    data = [word if word.isupper() and word.lower() in ner_tags else word.lower() for word in data]
    dollar_list = ['$','k','%']
    data = [d for d in data if len(d)>=2 or d in dollar_list]
#     clean_list.append(data)
    return data


# In[41]:


def __applyner(sequence):
    #########################################################################################
    # This method applies NER and returns the sequence according to the operation performed based on NER tag.
    #########################################################################################
    pickfile = open('/home/madhvi/IRE/MajorProject/Representations-in-Financial-Domain/tickermapping.pickle','rb')
    tickermapping = pickle.load(pickfile)
    ner_tags = []
    doc = nlp(sequence)  # applying NER
    for X in doc.ents:
        # If the NER class is ORG
        if X.label_ == 'ORG':
            "X.text can take microsoft corp or abcd name MSFT"
            text = X.text
            if text in tickermapping.keys():
                text = tickermapping[X.text]
            text = re.sub(r'[^\w\s]', '', X.text).lower()
            if 'inc' in text:
                text = text.replace('inc', '')
            if 'ltd' in text:
                text = text.replace('ltd', '')
            if 'llp' in text:
                text = text.replace('llp', '')
            if 'limited' in text:
                text = text.replace('limited', '')
            if 'corp' in text:
                text = text.replace('corp', '')
            if 'the' in text.lower():
                text = text.replace('the','')
            sequence = sequence.replace(X.text, text)
            ner_tags.extend(text.lower().split(" "))
        # If NER class is MONEY
        if X.label_ == 'MONEY':
            new_X = X.text.lower()
            if 'approximately' in new_X:  # Remove all the words which might appear in NER money class
                new_X = new_X.replace('approximately', '')
            if 'per' in new_X:
                new_X = new_X.replace('per', '')
            if 'to' in new_X:
                new_X = new_X.replace('to', '')
            if 'and' in new_X:
                new_X = new_X.replace('and', '')
            if 'between' in new_X:
                new_X = new_X.replace('between', '')
            if 'phone' in new_X:
                continue
            # Apply NER for the string which is obtained after removing other words this gives $200, $500 as separate ones
            if '$' not in new_X:
                new_X = "$"+new_X
            doc1 = nlp(new_X)
            for Y in doc1.ents:
                money = Y.text
                if ' ' not in money:
                    act_money = money.replace(',', '')  # Actual Money
                    #act_money = act_money.replace('.','')
                    sequence = sequence.replace(Y.text, act_money)  # Replace original money text with actual money
                    ner_tags.append(act_money)
                    # print(act_money)
                else:
                    money = Y.text[Y.text.find("$") + 1:]
                    k = money.find(' ')
                    try:
                        act_money = float(money[:k].replace(',', ''))
                        #act_money = act_money.replace('.','')
                        money_conv = w2n.word_to_num(money[k:])  # Conversion of word types million to *1e6
                        sequence = sequence.replace(Y.text, "$ "+str(act_money * money_conv))  # Replace original money text with actual money
                        #print("Converted from", money, act_money * money_conv)
                    except:
                        continue  # if any exception dont modify the original sentence and continue
        # If NER class is LAW
        if X.label_ == 'LAW':
            new_X = X.text
            new_X = re.sub(r'[\d.!?\-"]', '', new_X)
            if 'the' in new_X.lower():
                new_X = new_X.lower().replace('the', '')
            if 'of' in new_X.lower():
                new_X = new_X.lower().replace('of', '')
            if 'section' in new_X.lower():
                new_X = new_X.lower().replace('section', '')
            sequence = sequence.replace(X.text, new_X)
            ner_tags.extend(new_X.split(" "))
        # If NER class is Location
        if X.label_ == 'GPE':
            new_X = X.text.lower()
            new_X = re.sub(r'[\d.!?\-"]', '', new_X)
            if 'the' in new_X.lower():
                new_X = new_X.lower().replace('the', '')
            if '.' in new_X.lower():
                new_X = new_X.lower().replace('.', '')
            sequence = sequence.replace(X.text, new_X)
            ner_tags.extend(new_X.split(" "))
        # If NER class is Person
        if X.label_ == 'PERSON':
            new_X = X.text.lower()
            new_X = re.sub(r'[\d.!?\-"]', '', new_X)
            if 'the' in new_X.lower():
                new_X = new_X.lower().replace('the', '')
            if '.' in new_X.lower():
                new_X = new_X.lower().replace('.', '')
            sequence = sequence.replace(X.text, new_X)
            ner_tags.extend(new_X.split(" "))
        if X.label_ == 'CARDINAL':
            number = X.text
            number = number.replace(',','')
            #number = number.replace('.','')
            if number.isnumeric():
                sequence = sequence.replace(X.text, number)
        if X.label_ == 'QUANTITY':
            quantity = X.text.split(" ")
            for number in quantity:
                number = number.replace(',','')
                number = number.replace('.','')
                if number.isnumeric():
                    sequence = sequence.replace(X.text, number)
        if X.label_ == "PERCENT":
            percent = X.text.replace('%','')
            ner_tags.append(percent)
    return sequence,ner_tags


# In[42]:


def preprocess(sentences):
    cleaned_sentences = []
    for line in sentences:
        line = line.replace("$.","$0.")
        # Apply NER for the line
        line,ner_tags = __applyner(line)
        tokens = __clean_data(line,ner_tags)
        cleaned_sentences.append(tokens)
    return cleaned_sentences


# In[43]:


def get_embedding_dictionary(path, flag =True):
    if flag == True:
        pickfile = open(path,'rb')
        embeddings = pickle.load(pickfile)
        return embeddings
    print("Using Pre Trained Glove Embeddings")
    embeddings = {}
    glove = open(path,"r+")
    for line in glove:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings[word] = vector
    glove.close()
    return embeddings
    


# In[44]:


def get_embeddings(cleaned_sentences,embeddings):
#     missing_words = 0
#     total_words = 0
#     final_array = []
#     for sent in cleaned_sentences:
#         array = np.zeros((max_size, 300))
#         index=0
#         for word in sent:
#             total_words+=1
#             if word in embeddings.keys():
#                 array[index] = embeddings[word]
#     #             print(word, array[index])
#             else:
#     #             print(word)
#                 missing_words+=1
#             index+=1
#         final_array.append(array)
#     return np.asarray(final_array)
        
    vec = np.zeros([len(cleaned_sentences),300], dtype = 'float32') 
    c=0
    for i in cleaned_sentences:
        for j in i:
            try:
                j=str(j)
                k=embeddings[j]
                vec[c]=(vec[c]+np.array(k))
            except:
                continue
        c=c+1
    return vec


# In[45]:


cleaned_sentences = preprocess(sentences)


# In[46]:


max_size = 0
for cs in cleaned_sentences:
    max_size = max(max_size, len(cs))


# In[47]:


X_train, X_test, y_train, y_test = train_test_split(cleaned_sentences, scores, test_size=0.20, random_state=42)


# In[48]:


path = sys.argv[2]
embeddings = get_embedding_dictionary(path)
X_train = get_embeddings(X_train, embeddings)
X_test = get_embeddings(X_test, embeddings)


# In[49]:


x_val = X_train
x = np.array(x_val,dtype=np.float32)

y_val = y_train
y = np.array(y_val,dtype=np.float32)
y_tr = y.reshape(-1, 1)

x_val_t = X_test
x_test = np.array(x_val_t,dtype=np.float32)

y_val_t = y_test
y_ = np.array(y_val_t,dtype=np.float32)
y_t = y_.reshape(-1, 1)


# In[50]:


X_tr = torch.from_numpy(x.astype(np.float32))
Y_tr = torch.from_numpy(y_tr.astype(np.float32))

input_size = X_tr.shape[1]
output_size = 512;

model = nn.Sequential(
    nn.Linear(input_size, output_size),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(output_size, 128),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(64, 1))


learning_rate = 0.1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  



ep = []
tr_loss = []
num_epochs = 5000
for epoch in range(num_epochs):
    
    ep.append(epoch+1)
    y_pred = model(X_tr)
    loss = criterion(y_pred, Y_tr)
    tr_loss.append(loss.item())
    
    loss.backward()
    optimizer.step()


    optimizer.zero_grad()


    if (epoch+1)%10 == 0:
        print('epoch:', str(epoch+1), 'loss =', str(loss.item()))



plt.plot(ep,tr_loss,'r--')
plt.title('LOSS CURVE')
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.show()
torch.save(model, "Regression_model1")


# In[52]:


X_t = torch.from_numpy(x_test.astype(np.float32))
predicted = model(X_t).detach().numpy()


# In[55]:


print(mean_squared_error(y_t, predicted))

print(mean_absolute_error(y_t, predicted))

print(r2_score(y_t, predicted))


# In[56]:


path = sys.argv[4]
embeddings = get_embedding_dictionary(path, False)
X_train = get_embeddings(X_train, embeddings)
X_test = get_embeddings(X_test, embeddings)


# In[ ]:





# In[ ]:


x_val = X_train
x = np.array(x_val,dtype=np.float32)
print(x.shape)

y_val = y_train
y = np.array(y_val,dtype=np.float32)
y_tr = y.reshape(-1, 1)
print(y_tr.shape)

x_val_t = X_test
x_test = np.array(x_val_t,dtype=np.float32)
print(x_test.shape)

y_val_t = y_test
y_ = np.array(y_val_t,dtype=np.float32)
y_t = y_.reshape(-1, 1)
print(y_t.shape)


# In[58]:


X_tr = torch.from_numpy(x.astype(np.float32))
Y_tr = torch.from_numpy(y_tr.astype(np.float32))

input_size = X_tr.shape[1]
output_size = 512;

model = nn.Sequential(
    nn.Linear(input_size, output_size),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(output_size, 128),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.Dropout(0.5),
    nn.ReLU(),
    nn.Linear(64, 1))


learning_rate = 0.1
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  



ep = []
tr_loss = []
num_epochs = 5000
for epoch in range(num_epochs):
    
    ep.append(epoch+1)
    y_pred = model(X_tr)
    loss = criterion(y_pred, Y_tr)
    tr_loss.append(loss.item())
    
    loss.backward()
    optimizer.step()


    optimizer.zero_grad()


    if (epoch+1)%10 == 0:
        print('epoch:', str(epoch+1), 'loss =', str(loss.item()))



plt.plot(ep,tr_loss,'r--')
plt.title('LOSS CURVE')
plt.xlabel('Epochs')
plt.ylabel('Train Loss')
plt.show()


# In[60]:


X_t = torch.from_numpy(x_test.astype(np.float32))
predicted = model(X_t).detach().numpy()


# In[ ]:


print(mean_squared_error(y_t, predicted))

print(mean_absolute_error(y_t, predicted))

print(r2_score(y_t, predicted))

