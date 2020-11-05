###################################################################################################
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 22:05:51 2020

@author: Srikanth Thirumalasetti (Roll #2019900090)
"""

""" This file is part of Main project on 'Learning representations in Financial domain' """
""" It filters the list of files from the total corpus of SEC data by selecting recently filed SEC 
    data file for each US company. """
""" It copies the filtered list of files to the given output folder. """
###################################################################################################
import logging as log
import os
import pandas as pd
import shutil
import sys
import traceback
from dateutil.parser import parse
from glob import glob

def __parseSourceCorpusDocs(corpusDir):
    ######################################################################################
    # This method filters and generates a list of tuple that has the filtered file names.
    ######################################################################################
    try:
        docsListOfTuple = []
        docs = glob(os.path.join(corpusDir, "*.txt"))
        if not docs:
            log.error(f"There are NO original/pre-processed SEC text documents in the corpus folder '{corpusDir}'.")
        else:
            docCnt = 1
            for doc in docs:
                log.debug(f"Parsing file #{docCnt} of {len(docs)} files..")
                fileName = os.path.split(doc)[1]
                fileNameParts = fileName.split("_")
                if len(fileNameParts) != 5:
                    log.error(f"We parsed the file name '{fileName}'. But the format is different. Exiting out of this file without adding.")
                else:
                    datetime = parse(fileNameParts[1])
                    tplNew = (fileNameParts[0], datetime.year, fileName)
                    for i, tplOld in enumerate(docsListOfTuple):
                        if fileNameParts[0] in tplOld and datetime.year >= tplOld[1]:
                            docsListOfTuple[i] = tplNew
                    if not tplNew in docsListOfTuple:
                        docsListOfTuple.append(tplNew)
                docCnt += 1
        return docsListOfTuple
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = f"\n\t {exc_type}; {exc_value}"
        log.error(err)

def filterAndCopyCorpusFilesUsedToTrainBertOnMLM(corpusDir, outputFolderToCopyFilteredCorpusFiles):
    ######################################################################################
    # This method filters source corpus files of SEC data
    # and copies the filtered list of corpus of SEC files to the given output dir.
    ######################################################################################
    try:
        docsListOfTuple = __parseSourceCorpusDocs(corpusDir)
        if docsListOfTuple is None:
            log.error("Cannot get the list of tuple of source corpus files to copy.")
            return False

        singleCorpusTrainFile = os.path.join(corpusDir, "preProcessedCorpus.train")
        singleCorpusTestFile = os.path.join(corpusDir, "preProcessedCorpus.test")
        if os.path.exists(outputFolderToCopyFilteredCorpusFiles) is False:
            os.mkdir(outputFolderToCopyFilteredCorpusFiles)
            docCnt = 1
            with open(singleCorpusTrainFile, "a+", encoding="utf-8") as fTrain, open(singleCorpusTestFile, "a+", encoding="utf-8") as fTest:
                for i, tpl in enumerate(docsListOfTuple):
                    # Copy the file to the output dir
                    log.debug(f"Copying final training file #{i+1} of {len(docsListOfTuple)} files..")
                    shutil.copyfile(os.path.join(corpusDir, tpl[2]), os.path.join(outputFolderToCopyFilteredCorpusFiles, tpl[2]))
        return True
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = f"\n\t {exc_type}; {exc_value}"
        log.error(err)
