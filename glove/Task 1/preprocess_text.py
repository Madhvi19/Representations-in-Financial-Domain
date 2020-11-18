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

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
import re
from word2number import w2n
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn

def __expandContractions(sequence):
    #########################################################################################
    # This method expands contractions in the given sequence.
    #   For e.g., "didn't" will be converted to "did" and "not".
    #########################################################################################
    try:
        if sequence is not None and sequence.strip() != "":
            return contractions.fix(sequence)
        return sequence # return sequence as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while expanding contractions in the sequence '{0}'. Error is: {1}; {2}".format(sequence, str(exc_type), str(exc_value))
        raise Exception(err)

def __removeHtmlTags(sequence):
    #########################################################################################
    # This method removes any HTML tags and gets only the text from the given sequence.
    #   For e.g., in the sequence "<H1>This is the header</H1>", it removes H1 tag
    #   and returns "This is the header".
    #########################################################################################
    try:
        if sequence is not None and sequence.strip() != "":
            return re.sub(r'<(.*?)>','',sequence)
        return sequence # return sequence as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while removing HTML tags in the sequence '{0}'. Error is: {1}; {2}".format(sequence, str(exc_type), str(exc_value))
        raise Exception(err)

def __replaceurls(sequence):
    #########################################################################################
    # This method removes urls in the given sequence.
    #########################################################################################
    try:
        if sequence is not None and sequence.strip() != "":
            return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', sequence, flags=re.MULTILINE)
        return sequence # return sequence as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while removing HTML tags in the sequence '{0}'. Error is: {1}; {2}".format(sequence, str(exc_type), str(exc_value))
        raise Exception(err)

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
    data = __expandContractions(data)
    data = __removeHtmlTags(data) # remove tags
    data = __replaceurls(data)
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

def __applyner(sequence):
    #########################################################################################
    # This method applies NER and returns the sequence according to the operation performed based on NER tag.
    #########################################################################################
    pickfile = open(r'D:\College\Study\IRE\Project\Master\Representations-in-Financial-Domain\tickermapping.pickle','rb')
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

def preprocess_seq(sequence):
    try:
        sequence = sequence.replace("$.","$0.")
        line, ner_tags = __applyner(sequence)
        if len(line) > 0:
            tokens = __clean_data(line, ner_tags)
            if tokens:
                linestr = ""
                for token in tokens:
                    linestr += " " + token
                linestr = re.sub(r'\n+', '', linestr)
                linestr = re.sub(r'^\s*business', '', linestr)
                return linestr
        return sequence
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while pre-processing sequence. Error is: {0}; {1}".format( str(exc_type), str(exc_value))
        log.error(err)
        return sequence



































