#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
import re
from word2number import w2n
import os

from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional
from sklearn.model_selection import train_test_split


# In[3]:


def get_info(id, info_name, data_json):
    return data_json[id]['info'][0][info_name]


# In[4]:


def load_data(file_path):

    with open(file_path, 'r', encoding="utf-8") as file:
        tweets = []
        target_num = []
        offset = []
        target_cashtag = []
        relation = []

        data_json = json.load(file)

        def getAspects(aspect):
            aspect = aspect.replace('[', '')
            aspect = aspect.replace(']', '')
            aspect = aspect.replace('\'', '')
            return aspect.split('/')

        for id in data_json:
            tweets.append(id['tweet'].lower())
            target_num.append(id['target_num'])
            offset.append(id['offset'])
            target_cashtag.append(id['target_cashtag'].lower())
            relation.append(id['relation'])
            

    return tweets, target_num, offset, target_cashtag, relation


# In[5]:


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


# In[6]:


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


# In[7]:


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


# In[8]:


def preprocess(sentences):
    cleaned_sentences = []
    for line in sentences:
        line = line.replace("$.","$0.")
        # Apply NER for the line
        line,ner_tags = __applyner(line)
        tokens = __clean_data(line,ner_tags)
        cleaned_sentences.append(tokens)
    return cleaned_sentences


# In[9]:


def get_text(cleaned_tweets):
    text = []

    for sent in cleaned_tweets:
        s = ""
        for word in sent:
            s+= word+" "
        text.append(s.strip())
    return text


# In[2]:


embeddings = {}
glove = open(sys.argv[1],"r+")
for line in glove:
    values = line.split()
    word = values[0]
    vector = np.asarray(values[1:], dtype='float32')
    embeddings[word] = vector
glove.close()


# In[10]:


tweets, target_num, offset, target_cashtag, y_train = load_data(sys.argv[2])
cleaned_tweets_train = preprocess(tweets)
train_text = get_text(cleaned_tweets_train) 


# In[11]:


tweets, target_num, offset, target_cashtag, y_test = load_data(sys.argv[3])
cleaned_tweets_test = preprocess(tweets)
test_text = get_text(cleaned_tweets_test)


# In[12]:


vocab_size = 100000
oov_token = "<OOV>"
max_length = 300
padding_type = "post"
trunction_type="post"


# In[13]:


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(train_text)


# In[14]:


word_index = tokenizer.word_index
X_train_sequences = tokenizer.texts_to_sequences(train_text)
X_train_padded = pad_sequences(X_train_sequences,maxlen=max_length, padding=padding_type, 
                       truncating=trunction_type)


# In[15]:


X_test_sequences = tokenizer.texts_to_sequences(test_text)
X_test_padded = pad_sequences(X_test_sequences,maxlen=max_length, 
                               padding=padding_type, truncating=trunction_type)


# In[16]:


embedding_matrix = np.zeros((len(word_index) + 1, max_length))
for word, i in word_index.items():
    embedding_vector = embeddings.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


# In[17]:


embedding_layer = Embedding(len(word_index) + 1,
                            max_length,
                            weights=[embedding_matrix],
                            input_length=max_length,
                            trainable=False)
embedding_dim = 128
input_length = 100
model = Sequential([embedding_layer,
  Bidirectional(LSTM(embedding_dim, return_sequences=True, dropout=0.5)),
  Bidirectional(LSTM(embedding_dim, dropout = 0.25)),
  Dense(128, activation='relu'),
  Dense(64, activation='relu'),                  
  Dense(1, activation='sigmoid')
])
model.summary()


# In[18]:


model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# In[19]:


history = model.fit(X_train_padded, y_train, epochs=100, validation_data=(X_test_padded, y_test))


# In[ ]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# In[ ]:




