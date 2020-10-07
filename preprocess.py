#!/usr/bin/env python
# coding: utf-8

# In[1]:

import torchvision

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
import contractions
import sys
import logging as log
import pickle
nlp = en_core_web_sm.load()
from word2number import w2n
import os
import re
from Stemmer import Stemmer
from nltk.corpus import stopwords
import multiprocessing
import concurrent
import time
from nltk import PorterStemmer


global NUM_CPUs, MULTIPLY_FACTOR, INPUT_CORPUS_FILE
NUM_CPUs = multiprocessing.cpu_count()
MULTIPLY_FACTOR = 1

Stemmer = Stemmer('english')
stemmer = PorterStemmer()
docNames = {}
StopWords = set(stopwords.words("english"))
extension = set(["http", "https", "reflist", "yes","curlie","publish","page", "isbn", "file", "jpg", "websit", "cite", "title", "journal","publication", "name", "www","url","link", "ftp", "com", "net", "org", "archives", "pdf", "html", "png", "txt", "redirect", "align", "realign", "valign", "nonalign", "malign", "unalign", "salign", "qalign", "halign", "font", "fontsiz", "fontcolor", "backgroundcolor", "background", "style", "center", "text"])


pickfile = open('tickermapping.pickle','rb')
tickermapping = pickle.load(pickfile)

def setLogLevel(level):
    ###############################################################################################
    # This method sets the log level for the default logger.
    ###############################################################################################
    # Set log level, if set by the user
    # E for Error, D for Debug and I for Info
    if level == "I":
        log.basicConfig(level=log.INFO)
    elif level == "D":
        log.basicConfig(level=log.DEBUG)
    else:
        level = "E"
        log.basicConfig(level=log.ERROR)  # default to Error
    print("Setting log level to {0}".format(level))

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

def __removePunctuations(sequence):
    #########################################################################################
    # This method removes any punctuations and gets only the text from the given sequence.
    #########################################################################################
    try:
        if sequence is not None and sequence.strip() != "":
            return re.sub('[^A-Za-z0-9]+',' ',sequence)
        return sequence # return sequence as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while removing punctuations in the sequence '{0}'. Error is: {1}; {2}".format(sequence, str(exc_type), str(exc_value))
        raise Exception(err)

def __removeStopWords(sequence):
    #########################################################################################
    # This method removes stop words and gets only the remaining list from the given sequence.
    #########################################################################################
    try:
        return [word for word in sequence if word not in StopWords and word not in extension and bool(re.match('^(?=.*[a-zA-Z])(?=.*[0-9])', word)) ==False]
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while removing stop words in the sequence '{0}'. Error is: {1}; {2}".format(sequence, str(exc_type), str(exc_value))
        raise Exception(err)

def __stemList(sequence,ner_tags):
    #########################################################################################
    # This method gets stemmed words of the words in the sequence.
    #########################################################################################
    try:
        stemlist = []
        for word in sequence:
            if word in ner_tags:
                stemlist.append(word)
            else:
                stemlist.append(stemmer.stem(word))
        return stemlist
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while getitng stemmed list stop words in the sequence '{0}'. Error is: {1}; {2}".format(sequence, str(exc_type), str(exc_value))
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
    #     __removeStopWords
    #     __stemList
    #########################################################################################
    data = data.lower()
    data = __expandContractions(data)
    data = __removeHtmlTags(data) # remove tags
    data = __replaceurls(data)#re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', data, flags=re.MULTILINE) # remove url
    data = __removePunctuations(data)  # remove punctuations and special characters
    data = data.split(" ")
    data = __removeStopWords(data)#[word for word in data if word not in StopWords and word not in extension and bool(re.match('^(?=.*[a-zA-Z])(?=.*[0-9])', word)) ==False]
    data = __stemList(data,ner_tags)
    data = [d for d in data if len(d)>=2]
#     clean_list.append(data)
    return data

def __applyner(sequence):
    #########################################################################################
    # This method applies NER and returns the sequence according to the operation performed based on NER tag.
    #########################################################################################
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
            sequence = sequence.replace(X.text, text)
            ner_tags.extend(text.split(" "))
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
            doc1 = nlp(new_X)
            for Y in doc1.ents:
                money = Y.text[Y.text.find("$") + 1:]
                if ' ' not in money:
                    act_money = money.replace(',', '')  # Actual Money
                    sequence = sequence.replace(Y.text, act_money)  # Replace original money text with actual money
                    # print(act_money)
                else:
                    k = money.find(' ')
                    try:
                        act_money = float(money[:k].replace(',', ''))
                        money_conv = w2n.word_to_num(money[k:])  # Conversion of word types million to *1e6
                        sequence = sequence.replace(Y.text, act_money)  # Replace original money text with actual money
                        print("Converted from", money, act_money * money_conv)
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
    return sequence,ner_tags

def preprocess(directory_in_str,out_directory,docStartIndex,docEndIndex):
    global docNames
    directory = os.fsencode(directory_in_str)
    partialList = os.listdir(directory)[docStartIndex:docEndIndex]
    previous_line = ""
    for fil in partialList:
        filename = os.fsdecode(fil)
        cleanfile = open(out_directory+filename,'w')
        file = open(directory_in_str+filename,'r')
        HeaderFound = False
        for f in file:
            f = f[:-1]
            if("<FileName>" in f):
                print(f)
                docNames[filename] = (f.split('>')[1].split("<")[0])
            if "</Header>" in f:  #Read lines only after occurance of </Header>
                HeaderFound = True
                continue
            if HeaderFound==False:
                continue
            #Apply NER for the line
            f,ner_tags = __applyner(f)
            #To be changed: Start of preprocessing to sentences after applying NER
            if len(f)>0:
                tokens = __clean_data(f,ner_tags)
                if len(tokens)>0:
                    linestr = ""
                    for token in tokens:
                        linestr += token + " "
                    #print(linestr)
                    if previous_line != linestr:
                        previous_line = linestr
                        cleanfile.write(linestr+"\n")

        cleanfile.close()
    print("Writing to clean files is done by one processor")


if __name__ == "__main__":
    print("Total # of arguments passed to main() is {0}".format(len(sys.argv)))
    if len(sys.argv) < 3:
        print("** ERROR ** Corpus folder that has the list of 10k filings as documents AND folder to output final training corpus are required!")
        print("Usage:\n\t<this script> <corpus folder that has 10k files> <folder to output training corpus> <log level (E/D/I)>")
    else:
        _corpusFolder = sys.argv[1]
        _outFolder = sys.argv[2]
        if os.path.exists(_corpusFolder) is False:
            print("** ERROR ** Corpus folder '{0}' DOES NOT exist!".format(_corpusFolder))
            print("Usage:\n\t<this script> <corpus folder that has 10k files> <folder to output training corpus> <log level (E/D/I)>")
        else:
            if os.path.exists(_outFolder) is False:
                print("** ERROR ** Folder to output final training and evaluation corpus '{0}' DOES NOT exist!".format(_outFolder))
                print("Usage:\n\t<this script> <corpus folder that has 10k files> <folder to output training corpus> <log level (E/D/I)>")
            else:
                directory = os.fsencode(_corpusFolder)
                docs = os.listdir(directory)
                if not docs:
                    print("There are NO documents in the corpus folder '{0}'!".format(_corpusFolder))
                else:
                    level = "E"  # default to log.ERROR
                    if len(sys.argv) == 4:
                        logLevel = sys.argv[3]  # over default log level as set by the user
                    setLogLevel(logLevel)


                    totalDocs = len(docs)
                    docsRange = []
                    increments = totalDocs // NUM_CPUs  # this gives us how many documents can be processed by each processor
                    for i in range(0, totalDocs, increments):
                        docsRange.append(i)

                    startTime = time.time()
                    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPUs * MULTIPLY_FACTOR) as executor:
                        futures = []
                        log.info("Total # of documents in the corpus are: {0}".format(totalDocs))
                        log.info("Total # of CPUs in the system that this script is running (=# of processese that script runs in): {0}".format(NUM_CPUs))
                        for i in range(len(docsRange)):
                            if i < len(docsRange) - 1:
                                log.debug("Initializing a new process in the range: [{0}:{1}]".format(docsRange[i], docsRange[i+1]))
                                futures.append(executor.submit(preprocess, _corpusFolder, _outFolder, docsRange[i], docsRange[i+1]))
                            else:
                                log.debug("Initializing a new process in the range: [{0}:{1}]".format(docsRange[i], totalDocs))
                                futures.append(executor.submit(preprocess, _corpusFolder, _outFolder, docsRange[i], totalDocs))
                        try:
                            # Timeout after 1 hour
                            for future in concurrent.futures.as_completed(futures,3600):  # set timeout to 3600 seconds (1 hour)
                                success, startIndx, endIndx = future.result()
                                if success is not True:
                                    log.error("Error building output text corpus for the 10-k document range [{0}:{1}].".format(startIndx, endIndx))
                            log.info("\nIt took {0} minutes to build the output text corpus file from ALL the 10-k documents in the corpus folder '{1}'".format(round((time.time() - startTime)/60, 0), _corpusFolder))
                        except:
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            log.error("Error (timeout or other) has occurred while building output text corpus file for training. Error is: {0}, {1}".format(exc_type, exc_value))
