###################################################################################################
###################################################################################################
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 22:05:51 2020

@author: Srikanth Thirumalasetti (Roll #2019900090)
"""

""" This file is part of Main project on 'Learning representations in Financial domain' """

""" It generates text corpus from SEC 10k filings that are used as input data to fine-tune/train a language model on MLM task (BERT model). """

""" This script does the following:
    1. Reads the financial text corpus from the given 33,612 files (SEC's 10-k documents) for training LM model.
    2. Pre-processes and tokenizes sentences in all the documents.
    3. Creates two text files used as corpus for training and evaluating our language model: 
        <out folder>/sec_10k_docs.textcorpus.trg
        <out folder>/sec_10k_docs.textcorpus.tst
"""

""" This script can run on multiple-processors. 
    It automatically sets the total # of processors (based on the system it is running) to build the output text corpus for training and evaluation.
"""

###################################################################################################
import concurrent.futures
import glob
import logging as log
import math
import multiprocessing
import os
import re
import sys
import threading
import time
import traceback
from collections import defaultdict, OrderedDict, deque
from functools import reduce
from itertools import islice
from nltk.corpus import stopwords as sw
from nltk.tokenize import sent_tokenize, word_tokenize
from textPreProcessing import cleanTextSequence

# Global variables that are accessed by ALL the processors
global NUM_CPUs, MULTIPLY_FACTOR, INPUT_CORPUS_FILE
NUM_CPUs = multiprocessing.cpu_count()
MULTIPLY_FACTOR = 1
OUTPUT_CORPUS_FILE_TRG = "sec_10k_docs.textcorpus.trg"  # this is our output file generated for training
OUTPUT_CORPUS_FILE_TST = "sec_10k_docs.textcorpus.tst"  # this is our output file generated for testing

###############################################
# This class returns responses to the queries.
###############################################
class BuildTextCorpusForTraining:
    def __init__(self, logLevel, corpusFolder, outFolder):
        #########################################################################################
        # This method is class initialzer that inits several instance level variables,
        # including global lock object from the concurrent.features manager object.
        #########################################################################################

        # Set the log level for logger in the class also, as when a user warning is thrown by
        # any of the packages, the logger's log level is getting reset to WARN, which means that
        # INFO and DEBUG statements are not printed to stdout or a log file.
        setLogLevel(logLevel)

        log.info("Initializing 'buildTextCorpusForTraining' class instance for a new process..")

        # Variable that holds the path to the folder that has our 10-k files
        self.corpusFolder = corpusFolder

        # Running count of total sentences in the corpus
        self.totalSentencesInCorpus = 0

        # List that has total sentences for every increment of DOCS_COUNT_TO_WRITE_FILE
        self.tokenizedSentences = []

        # This is our output file generated for training and testing when this script executes successfully.
        # These two files are fed as input for training our BERT language model with custom 10-k sec filings data.
        self.OUTPUT_CORPUS_FILE_TRG = OUTPUT_CORPUS_FILE_TRG  # this is our output file generated for training
        self.OUTPUT_CORPUS_FILE_TST = OUTPUT_CORPUS_FILE_TST  # this is our output file generated for testing
        if outFolder is not None and outFolder != "" and os.path.exists(outFolder):
            self.OUTPUT_CORPUS_FILE_TRG = os.path.join(outFolder, OUTPUT_CORPUS_FILE_TRG)  # this is our output file generated for training
            self.OUTPUT_CORPUS_FILE_TST = os.path.join(outFolder, OUTPUT_CORPUS_FILE_TST)  # this is our output file generated for testing

        # Variable that holds the max paragraph length the is fed to our BERT language model for training.
        # For BERT model training, we need to input max_seq_length < 512 (total word pieces and not total words/ tokens)
        self.MAX_PARA_LENGTH = 480

        # Variable that tells when to write the tokenized sentences to our OUTPUT_CORPUS_FILE_TRG
        # This value is per CPU or per document range.
        self.DOCS_COUNT_TO_WRITE_TO_FILE = 250

        # List of stop words derived from NLTK + our own set of stop words
        self.stopWordsDict = defaultdict(int)
        for word in set(sw.words("english")):
            self.stopWordsDict[word] = 1
        # Our own set of stop words
        additionalStopWords = { "<" : 1, ">" : 1, "(" : 1, ")" : 1, "\\" : 1, "/" : 1, "--": 1, "..." : 1, "``" : 1, "''" : 1, "'s" : 1, "**" : 1 }
        self.stopWordsDict.update(additionalStopWords)

        # Global lock used while building OUTPUT_CORPUS_FILE_TRG
        self.lock = threading.Lock()

        # Variable that is used to print a new blank like for each doc that starts with this text
        self.NEW_DOCUMENT_HEADER_USED_TO_START_NEW_LINE = "NEW_DOCUMENT_STARTS_FROM_HERE"

    def __preProcessSentence(self, sent):
        ########################################################################################
        # This method pre-processes the given sequence. For now, it just applies NLTK tokenizer.
        ########################################################################################
        try:
            if sent is None or sent == "":
                return
            # terms = word_tokenize(sent)
            # if terms:
            #     return [term for term in terms if term not in self.stopWordsDict]
            return cleanTextSequence(sent).split()
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = "Error occurred while pre-processing the sequence '{0}'. Error is: {1}; {2}".format(sent,str(exc_type)),str(exc_value)
            raise Exception(err)

    def __generateTokenizedSentences(self, doc):
        #########################################################################################
        # This method generates tokenized sentences from text in the given document. A tokenized
        # sentence is built as follows:
        #   1. Read all the lines in the document
        #   2. Call the relevant preprocessing method to generate tokenized sentences
        #   3. For each tokenized sentences, split in whitespace and tokenize each word
        #   4. Join the tokens in each sentences and return the tokenizedSentence as a list().
        #########################################################################################
        try:
            if doc is None or doc == "" or os.path.exists(doc) is False:
                log.error("Document '{0}' DOES NOT exist!".format(doc))
                return
            else:
                log.debug("Document '{0}' exists! Started processing the sentences in the document..".format(doc))

            sentences = []
            sentencesTokenized = []
            with open(doc,'r',encoding='utf-8') as f:
                # Tokenize sentences in the file
                lines = []
                for line in f:
                    # Read text starting after </header> as being done in pre-processing method
                    line = line.strip()
                    if line is not None and line != "":
                        lines.append(line)
                log.debug("There are a total {0} lines in the document '{1}'..".format(len(lines),doc))
                log.debug("Started tokenizing the sentences by reading ALL the lines in the document '{0}'..".format(doc))
                sentences = sent_tokenize(" ".join(lines))
                log.debug("Finished tokenizing the sentences by reading ALL the lines in the document '{0}'.".format(doc))
                log.debug("Total # of sentences in the document '{0}' that are being processed are {1}.".format(doc, len(sentences)))
                if sentences:
                    for sentence in sentences:
                        # Pre-process each sentence
                        tokens = self.__preProcessSentence(sentence)
                        if tokens:
                            sentencesTokenized.append(" ".join(tokens))
            return sentencesTokenized
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = "Error occurred while generating tokens for the document '{0}'. Error is: {1}; {2}".format(doc,str(exc_type)), str(exc_value)
            raise Exception(err)

    def __writeToFile(self, toTrainingFile):
        ########################################################################################
        # This method writes the tokenized sentences from each document to the OUTPUT_CORPUS_FILE_TRG.
        ########################################################################################
        try:
            fil = self.OUTPUT_CORPUS_FILE_TRG # default to training corpus
            if toTrainingFile == "TST":
                fil = self.OUTPUT_CORPUS_FILE_TST # write to test corpus

            if self.tokenizedSentences:
                self.lock.acquire()
                with open(fil, "a+", encoding='utf-8', errors='ignore') as f:
                    para = ""
                    for tknzdSntc in self.tokenizedSentences:
                        tknzdSntc = tknzdSntc.strip()
                        # Build a paragraph by concatenating tokenized sentences such that:
                        # the total length of the paragraph is < self.MAX_PARA_LENGTH.
                        if tknzdSntc.lower().find(self.NEW_DOCUMENT_HEADER_USED_TO_START_NEW_LINE.lower(), 0) > -1:
                            # print blank lines as this is the header text of a new document
                            para = "{0}\n\n{1}".format(para, tknzdSntc)
                        else:
                            para = "{0} {1}".format(para, tknzdSntc)
                        if len(para) > self.MAX_PARA_LENGTH:
                            para = para.replace(self.NEW_DOCUMENT_HEADER_USED_TO_START_NEW_LINE, "")
                            f.write(para.strip() + "\n\n")
                            para = ""
                self.lock.release()
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = "Error occurred while writing tokenized sentences to the file '{0}'. Error is: {1}; {2}".format(self.OUTPUT_CORPUS_FILE_TRG,str(exc_type)), str(exc_value)
            raise Exception(err)

    def buildInputCorpusFile(self, startDocIndex, endDocIndex):
        #########################################################################################
        # This method builds text corpus in a single file as follows:
        #   1. It reads each document in the text corpus folder.
        #   2. It generates tokenized sentences for each document.
        #   3. The tokenized sentences are appended by following the rules for input as
        #       mentioned in the file comments in the file header above.
        #   4. Finally, it writes the global inputs list to the files: <out folder>/sec_10k_docs.textcorpus.*
        #########################################################################################
        startTime = time.time()
        try:
            if self.corpusFolder is None or self.corpusFolder == "" or os.path.exists(self.corpusFolder) is False:
                raise Exception("** ERROR ** Text corpus folder '{0}' DOES NOT exist!".format(self.corpusFolder))

            log.info("Started building output text corpus for training from the 10-k documents in the corpus folder ranging from [{0}:{1}] at: {2}.".format(startDocIndex, endDocIndex, time.asctime(time.localtime())))

            # Get the list of documents in the corpus folder and loop through each document to generate tokens
            docs = glob.glob(os.path.join(self.corpusFolder, "*.txt"))
            if not docs:
                raise Exception("There are NO documents in the corpus folder '{0}'.".format(self.corpusFolder))

            docs = docs[startDocIndex:endDocIndex]
            log.debug("Total {0} documents in the range [{1}:{2}] are being processed..".format(len(docs), startDocIndex, endDocIndex))
            docCnt = 0              # running count of documents that were processed thus far
            docNumIncrements = 0    # integer value that holds the running multiples of DOCS_COUNT_TO_WRITE_TO_FILE
            for doc in docs:
                # Build the inputs and append to the global inputs list
                log.debug("Started pre-processing to generate tokenized sentences for the document '{0}'..".format(doc))
                tokenizedSentences = self.__generateTokenizedSentences(doc)
                log.debug("Finished pre-processing and generated tokenized sentences for the document '{0}'.".format(doc))
                if tokenizedSentences:
                    docCnt += 1
                    # Update the global total # of sentences for the next document
                    self.totalSentencesInCorpus += len(tokenizedSentences)
                    log.info("Total # of documents processed thus far in the 10-k documents range [{0}:{1}] are {2} and Total # of sentences tokenized thus far are '{3}'.".format(startDocIndex, endDocIndex, (docCnt + (self.DOCS_COUNT_TO_WRITE_TO_FILE * docNumIncrements)), self.totalSentencesInCorpus))
                    tokenizedSentences.insert(0, self.NEW_DOCUMENT_HEADER_USED_TO_START_NEW_LINE)
                    self.tokenizedSentences.extend(tokenizedSentences)
                    if docCnt == self.DOCS_COUNT_TO_WRITE_TO_FILE:
                        if docNumIncrements > 0 and docNumIncrements % 4 == 0: # 25% of documents go into test corpus
                            self.__writeToFile("TST") # write to test file
                        else:
                            self.__writeToFile("TRG") # write to training file
                        docNumIncrements += 1
                        docCnt = 0  # reset
                        self.tokenizedSentences = [] # reset

            # Write any residual tokenized sentences that do not add up to DOCS_COUNT_TO_WRITE_TO_FILE
            self.__writeToFile("TRG")

            log.info("Finished generating output text corpus file from 10-k documents ranging from [{0}:{1}]. There are a total of {2} sentences that are tokenized from the 10-k documents in this range.".format(startDocIndex, endDocIndex, self.totalSentencesInCorpus))
            log.info("It took {0} minutes to build the output text corpus '{1}' from 10-k documents ranging from [{2}:{3}]".format(round((time.time() - startTime)/60,0), self.OUTPUT_CORPUS_FILE_TRG, startDocIndex, endDocIndex))
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = "Error occurred while generating output text corpus file by tokenizing sentences in ALL the documents in the text corpus folder '{0}'. Error is: {1}; {2}".format(self.corpusFolder,str(exc_type)),str(exc_value)
            log.exception(err)

def buildTextCorpusPartial(logLevel, corpusFolder, outFolder, startDocIndex, endDocIndex):
    ###############################################################################################
    # This method is called by ProcessPoolExecutor to create individual class instances to process
    # respective documents in the given range.
    ###############################################################################################
    try:
        cls = BuildTextCorpusForTraining(logLevel, corpusFolder, outFolder)

        # Get the list of documents in the corpus folder and loop through each document to generate tokens
        docs = glob.glob(os.path.join(corpusFolder, "*.txt"))
        if not docs:
            log.error("There are NO documents in the corpus folder '{0}'!".format(corpusFolder))

        try:
            cls.buildInputCorpusFile(startDocIndex, endDocIndex)
            if os.path.exists(cls.OUTPUT_CORPUS_FILE_TRG):
                log.info("Successfully built output text corpus '{0}' from 10-k documents in the corpus folder ranging from [{1}:{2}] to train BERT language model using SEC 10k filings data.".format(cls.OUTPUT_CORPUS_FILE_TRG, startDocIndex, endDocIndex))
                cls = None
                return (True, startDocIndex, endDocIndex)
            cls = None
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = "Error occurred in 'buildInputCorpusFile' method. Error is: {2}; {3}".format(startDocIndex, endDocIndex, exc_type, exc_value)
            log.exception(err)

        return (False, startDocIndex, endDocIndex)
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while generating output text corpus file for the 10-k documents ranging from [{0}:{1}]. Error is: {2}; {3}".format(startDocIndex, endDocIndex, exc_type, exc_value)
        log.exception(err)

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
                level = "E" # default to log.ERROR
                if len(sys.argv) == 4:
                    logLevel = sys.argv[3]  # over default log level as set by the user
                setLogLevel(logLevel)

                # Delete OUTPUT_CORPUS_FILE_TRG and OUTPUT_CORPUS_FILE_TST if already exists
                if os.path.exists(os.path.join(_outFolder, OUTPUT_CORPUS_FILE_TRG)):
                    os.remove(os.path.join(_outFolder, OUTPUT_CORPUS_FILE_TRG))
                if os.path.exists(os.path.join(_outFolder, OUTPUT_CORPUS_FILE_TST)):
                    os.remove(os.path.join(_outFolder, OUTPUT_CORPUS_FILE_TST))

                docs = glob.glob(os.path.join(_corpusFolder, "*.txt"))
                if not docs:
                    log.error("There are NO documents in the corpus folder '{0}'!".format(_corpusFolder))
                else:
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
                                futures.append(executor.submit(buildTextCorpusPartial, logLevel, _corpusFolder, _outFolder, docsRange[i], docsRange[i+1]))
                            else:
                                log.debug("Initializing a new process in the range: [{0}:{1}]".format(docsRange[i], totalDocs))
                                futures.append(executor.submit(buildTextCorpusPartial, logLevel, _corpusFolder, _outFolder, docsRange[i], totalDocs))
                        try:
                            # Timeout after 600 seconds
                            for future in concurrent.futures.as_completed(futures,1200):  # set timeout to 1200 seconds (20 minutes)
                                success, startIndx, endIndx = future.result()
                                if success is not True:
                                    log.error("Error building output text corpus for the 10-k document range [{0}:{1}].".format(startIndx, endIndx))
                            log.info("\nIt took {0} minutes to build the output text corpus file from ALL the 10-k documents in the corpus folder '{1}'".format(round((time.time() - startTime)/60, 0), _corpusFolder))
                        except:
                            exc_type, exc_value, exc_traceback = sys.exc_info()
                            log.error("Error (timeout or other) has occurred while building output text corpus file for training. Error is: {0}, {1}".format(exc_type, exc_value))
