#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
import re
from word2number import w2n
import os
import re
from Stemmer import Stemmer
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords


# In[2]:


directory_in_str = "/home/madhvi/IRE/MajorProject/10k_1900_org_sample/" #Directory where all the files are stored
directory = os.fsencode(directory_in_str)

Stemmer = Stemmer('english')
docNames = {}
StopWords = set(stopwords.words("english"))
extension = set(["http", "https", "reflist", "yes","curlie","publish","page", "isbn", "file", "jpg", "websit", "cite", "title", "journal","publication", "name", "www","url","link", "ftp", "com", "net", "org", "archives", "pdf", "html", "png", "txt", "redirect", "align", "realign", "valign", "nonalign", "malign", "unalign", "salign", "qalign", "halign", "font", "fontsiz", "fontcolor", "backgroundcolor", "background", "style", "center", "text"])


# In[3]:


def clean_data(data):
    data = data.lower()
    data = re.sub(r'<(.*?)>','',data) # remove tags
    data = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', data, flags=re.MULTILINE) # remove url 
    data = re.sub('[^A-Za-z0-9]+',' ',data)  # remove punctuations and special characters
#         data = word_tokenize///(data)                         # tokenize string
    data = data.split(" ")
    data = [word for word in data if word not in StopWords and word not in extension and bool(re.match('^(?=.*[a-zA-Z])(?=.*[0-9])', word)) ==False] 
    data = [Stemmer.stemWord(word) for word in data]

#     clean_list.append(data)
    return data


# In[4]:


def preprocess():
    global docNames
    for fil in os.listdir(directory):
        flist = []
        filename = os.fsdecode(fil)
        file = open("/home/madhvi/IRE/MajorProject/10k_1900_org_sample/"+filename,'r')
        ak = False
        for f in file:
            f = f[:-1]
            if("<FileName>" in f):
                print(f)
                docNames[filename] = (f.split('>')[1].split("<")[0])
            if "</Header>" in f:  #Read lines only after occurance of </Header>
                ak = True
                continue
            if ak==False:
                continue
            doc= nlp(f) #applying NER
            nermap = {} # Map to hold mapping from NER applied tokens to original text
            for X in doc.ents:
                if X.label_=='ORG' or X.label_=='PERSON':    # If the NER class is ORG or PERSON
                    text = re.sub(r'[^\w\s]', '', X.text)
                    text = text.replace(" ","")
                    f = f.replace(X.text,text)
                    nermap[text] = X.text
                if  X.label_ == 'MONEY':   #If NER class is MONEY
                    new_X = X.text
                    if 'approximately' in new_X.lower():    #Remove all the words which might appear in NER money class
                        new_X = new_X.lower().replace('approximately','')
                    if 'per' in new_X.lower():
                        new_X = new_X.lower().replace('per','')
                    if 'to' in new_X.lower():
                        new_X = new_X.lower().replace('to','')
                    if 'and' in new_X.lower():
                        new_X = new_X.lower().replace('and','')
                    if 'between' in new_X.lower():
                        new_X = new_X.lower().replace('between','')
                    if 'phone' in new_X.lower():
                        continue
                    doc1 = nlp(new_X) #Apply NER for the string which is obtained after removing other words this gives $200, $500 as separate ones
                    for Y in doc1.ents:
                        money = Y.text[Y.text.find("$")+1:]
                        if ' ' not in money:
                            act_money = money.replace(',','')   #Actual Money
                            f = f.replace(Y.text,act_money)   #Replace original money text with actual money
                            #print(act_money)
                        else:
                            k = money.find(' ')
                            try:
                                act_money = float(money[:k].replace(',',''))
                                money_conv = w2n.word_to_num(money[k:]) #Conversion of word types million to *1e6
                                f = f.replace(Y.text,act_money) #Replace original money text with actual money
                                print("Converted from",money,act_money*money_conv)
                            except:
                                continue # if any exception dont modify the original sentence and continue
                if  X.label_ == 'LAW':   
                    new_X = X.text
                    new_X = re.sub(r'[\d.!?\-"]', '', new_X)
                    if 'the' in new_X.lower():    
                        new_X = new_X.lower().replace('the','')
                    if 'of' in new_X.lower():
                        new_X = new_X.lower().replace('of','')
                    if 'section' in new_X.lower():
                        new_X = new_X.lower().replace('section','')
                    new_X = new_X.replace(" ","")
                    f = f.replace(X.text, new_X)
                    nermap[new_X] = X.text
                if X.label_ == 'GPE':
                    new_X = X.text.lower()
                    new_X = re.sub(r'[\d.!?\-"]', '', new_X)
                    if 'the' in new_X.lower(): 
                        new_X = new_X.lower().replace('the','')
                    if '.' in new_X.lower():
                        new_X = new_X.lower().replace('.','')
                    new_X = new_X.replace(" ","")
                    f = f.replace(X.text, new_X)
                    #print(new_X, X.label_)
                    nermap[new_X] = X.text
                if X.label_ == 'PERSON':
                    new_X = X.text.lower()
                    new_X = re.sub(r'[\d.!?/\-"]', '', new_X)
                    new_X = re.sub(r'\s', '', new_X)
                    new_X = re.sub(r's$', '', new_X)
                    if '.' in new_X.lower():
                        new_X = new_X.lower().replace('.','')
                    new_X = new_X.replace(" ","")
                    f = f.replace(X.text, new_X)
                    #print(new_X, X.label_)
                    nermap[new_X] = X.text
            #To be changed: Start of preprocessing to sentences after applying NER
            if len(f)>0:
                fl = clean_data(f)
#                 fl = f.split(" ")
                for fi in fl:
                    if fi in nermap.keys():
                        flist.append(nermap[fi])
                    else:
                        if len(fi)>2:
                            flist.append(fi)

            
        print(flist)    #To be added: Write to a file
        break


# In[5]:


preprocess()


# In[6]:


docNames


# In[ ]:




