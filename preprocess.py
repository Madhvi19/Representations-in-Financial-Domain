import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
import re
from word2number import w2n
import os


directory_in_str = "/Users/skosgi/Downloads/10k_1900_org_sample/" #Directory where all the files are stored
directory = os.fsencode(directory_in_str)



def preprocess():
    for fil in os.listdir(directory):

        flist = []
        filename = os.fsdecode(fil)
        file = open("/Users/skosgi/Downloads/10k_1900_org_sample/"+filename,'r')
        ak = False
        for f in file:
            f = f[:-1]
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
            #To be changed: Start of preprocessing to sentences after applying NER
            fl = f.split(" ")
            for fi in fl:
                if fi in nermap.keys():
                    flist.append(nermap[fi])
                else:
                    if len(fi)>0:
                        flist.append(fi)

        print(flist)    #To be added: Write to a file




preprocess()