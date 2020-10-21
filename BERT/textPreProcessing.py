###################################################################################################
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 22:05:51 2020

@author: Srikanth Thirumalasetti (Roll #2019900090)
"""

""" This file is part of Main project on 'Learning representations in Financial domain' """

""" It has methods to pre-process the given sequence or list of sequences or complete text in the given document. """
###################################################################################################
import contractions
import inflect
import os
import re
import string
import sys
import time
import traceback
import unicodedata as ucd
from bs4 import BeautifulSoup as bs
from collections import defaultdict, OrderedDict
from nltk.corpus import stopwords as sw
from nltk.corpus.reader.wordnet import NOUN, VERB, ADJ, ADV
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import LancasterStemmer, WordNetLemmatizer

stopWordsDict = defaultdict(int)
for word in set(sw.words("english")):
    stopWordsDict[word] = 1
# Our own set of stop words
additionalStopWords = {"<": 1, ">": 1, "(": 1, ")": 1, "\\": 1, "/": 1, "--": 1, "...": 1, "``": 1, "''": 1, "'s": 1, "**": 1, ":" : 1}
stopWordsDict.update(additionalStopWords)

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
            bsoup = bs(sequence, "html.parser")
            return bsoup.get_text()
        return sequence # return sequence as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while removing HTML tags in the sequence '{0}'. Error is: {1}; {2}".format(sequence, str(exc_type), str(exc_value))
        raise Exception(err)

def __replaceTextUsingRegEx(sequence, regEx, textToReplace):
    #########################################################################################
    # This method applies the given regular expression on the given sequence.
    #########################################################################################
    try:
        if sequence is not None and sequence.strip() != "":
            return re.sub(regEx, textToReplace, sequence)
        return sequence # return sequence as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while replacing text '{0}' in the sequence '{1}' by applying the regular expression '{2}'. Error is: {3}; {4}".format(textToReplace, sequence, regEx, str(exc_type), str(exc_value))
        raise Exception(err)

def __convertToAscii(words):
    #########################################################################################
    # This method converts the given words to their ASCII equivalent.
    #########################################################################################
    try:
        if words:
            # NFKD is normal form D with compatability decompositon, which means replace all characters with their equivalents.
            # For e.g, Roman number I and alphabet I.
            return [ucd.normalize("NFKD", word).encode("ascii", "ignore").decode("utf-8","ignore") for word in words if word != ""]
        return words # return words as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while converting the words '{0}' to ASCII. Error is: {1}; {2}".format(" ".join(words), str(exc_type), str(exc_value))
        raise Exception(err)

def __convertToLowerCase(words):
    #########################################################################################
    # This method converts the given words to lower case.
    #########################################################################################
    try:
        if words:
            return [word.lower() for word in words if word != "" and word.isdigit() is False]
        return words # return words as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while converting the words '{0}' to lowercase. Error is: {1}; {2}".format(" ".join(words), str(exc_type), str(exc_value))
        raise Exception(err)

def __getStems(words):
    #########################################################################################
    # This method returns stemmed words by applying nltk.LancasterStemmer.
    #########################################################################################
    try:
        if words:
            stemmer = LancasterStemmer()
            return [stemmer.stem(word) for word in words if word != ""]
        return words # return words as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while getting stems of the words '{0}'. Error is: {1}; {2}".format(" ".join(words), str(exc_type), str(exc_value))
        raise Exception(err)

def __getLemmas(words):
    #########################################################################################
    # This method returns lemmas of words that are verbs by applying nltk.WordNetLemmatizer.
    #########################################################################################
    try:
        if words:
            lemmatizer = WordNetLemmatizer()
            return [lemmatizer.lemmatize(word, pos=VERB) for word in words if word != ""]
        return words # return words as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while getting lemmas of the words '{0}'. Error is: {1}; {2}".format(" ".join(words), str(exc_type), str(exc_value))
        raise Exception(err)

def __convertNumbersToText(words):
    #########################################################################################
    # This method converts numbers in the given words to text.
    #########################################################################################
    try:
        if words:
            inf = inflect.engine()
            return [inf.number_to_words(word) if word.isdigit() else word for word in words]
        return words # return words as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while converting converting numbers to text in the words '{0}'. Error is: {1}; {2}".format(" ".join(words), str(exc_type), str(exc_value))
        raise Exception(err)

def __removeStopWords(words):
    #########################################################################################
    # This method removes the stop words from the given words.
    #########################################################################################
    try:
        if words:
            return [word for word in words if word not in stopWordsDict]
        return words # return words as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while removing stop words from the words '{0}'. Error is: {1}; {2}".format(" ".join(words), str(exc_type), str(exc_value))
        raise Exception(err)

def __removeSingleAndDuplicates(words):
    #########################################################################################
    # This method removes words of length 1 and duplicates.
    #########################################################################################
    try:
        uniqueWords = []
        if words:
            for word in words:
                if not uniqueWords.__contains__(word) and len(word) > 1:
                    uniqueWords.append(word)
            return uniqueWords
        return words # return words as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while converting the words '{0}' to ASCII. Error is: {1}; {2}".format(" ".join(words), str(exc_type), str(exc_value))
        raise Exception(err)

def cleanTextSequence(sequence, convertNumbersToText = False, removeHtmlTags = True, regExToFindAndReplace = "", textToReplaceInRegExFind = ""):
    #########################################################################################
    # This method applies the following methods in the given sequence and returns a clean and noise-less sequence.
    # The below methods are being applied in the order they are listed here:
    #       __expandContractions(sequence)
    #       __removeHtmlTags(sequence)
    #       __replaceTextUsingRegEx(sequence)
    #       word_tokenize(sequence)
    #       __convertToAscii(words)
    #       __convertToLowerCase(words)
    #       __getStems(words)
    #       __getLemmas(words)
    #       __convertNumbersToText(words)
    #       __removeStopWords(words)
    #       __removeSingleAndDuplicates(words)
    #########################################################################################
    try:
        if sequence is not None and sequence.strip() != "":
            # @ sequence level
            cleanedSequence = __expandContractions(sequence)
            if removeHtmlTags is True:
                cleanedSequence = __removeHtmlTags(cleanedSequence)
            if regExToFindAndReplace is not  None and regExToFindAndReplace != "":
                cleanedSequence = __replaceTextUsingRegEx(cleanedSequence, textToReplaceInRegExFind)

            # @ words level
            if cleanedSequence.strip() != "":
                words = word_tokenize(cleanedSequence)
                words = __convertToLowerCase(__convertToAscii(words))
                words = __getStems(words)
                words = __getLemmas(words)
                if convertNumbersToText is True:
                    words = __convertNumbersToText(words)
                words = __removeSingleAndDuplicates(__removeStopWords(words))
                sequence = " ".join(words)

        return sequence  # return sequence as is without any changes
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while tokenizing and normalizing text in the sequence '{0}'. Error is: {1}; {2}".format(sequence, str(exc_type), str(exc_value))
        raise Exception(err)

def cleanTextSequences(listOfSequences):
    #########################################################################################
    # This method tokenizes, normalizes and removes noise in each sequence in the list of given sequences.
    #########################################################################################
    try:
        cleanedList = []
        if listOfSequences:
            for sequence in listOfSequences:
                cleanedList.append(cleanTextSequence(sequence))
        return cleanedList
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while tokenizing and normalizing text the list of sequences starting with the sequence '{0}'. Error is: {1}; {2}".format(listOfSequences[0], str(exc_type), str(exc_value))
        raise Exception(err)

def cleanTextInDocument(doc):
    #########################################################################################
    # This method tokenizes, normalizes and removes noise in the document and returns a clean
    # document with the same name + ".clean" appended to the file name and saved in the same folder.
    #########################################################################################
    try:
        if os.path.exists(doc):
            docCleaned = os.path.join(os.path.split(doc)[0], os.path.split(doc)[1] + ".clean")
            with open(doc, 'r', encoding='utf-8') as f:
                # Read all the lines in the file
                lines = []
                for line in f:
                    # Remove \n char
                    line = line.strip()
                    # Do not include single alphabet or digit line (for e.g., '1' or 'a' etc..)
                    if line is not None and line != "" and len(line.split()) > 1:
                        lines.append(line)

                # Pre-process each line
                cleanedSequencesInAllLines = cleanTextSequences(lines)
                if cleanedSequencesInAllLines:
                    with open(docCleaned, 'w+', encoding='utf-8') as f:
                        f.writelines(cleanedSequencesInAllLines)

            # Finally, return if the document is successfully cleaned and a new document is built with ".clean" appended to the original file name
            if os.path.exists(docCleaned):
                return docCleaned
            else:
                return None
        return None
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = "Error occurred while tokenizing and normalizing text in the document '{0}'. Error is: {1}; {2}".format(doc, str(exc_type), str(exc_value))
        raise Exception(err)
