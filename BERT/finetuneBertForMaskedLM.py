###################################################################################################
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 22:05:51 2020

@author: Srikanth Thirumalasetti (Roll #2019900090)
"""

""" This file is part of Main project on 'Learning representations in Financial domain' """
""" It finetunes and evaluates a pre-trained BERT (large-uncased) model withe same training objective i.e. MLM. 
    It uses the text corpus derived from 10-k filings with SEC) """
""" BERT embeddings for the financial domain are generated as follows:
    1. Install Simple transformers on a single GPU system.
    2. Collate text files for training and evaluation with max seq length as given by the user.
    3. Use simple transformer's 'LanguageModelingModel' class to train and evaluate a pre-trained BERT model with our custom SEC data.
"""
###################################################################################################
import glob
import logging as log
import math
import multiprocessing
import os
import re
import sys
import threading
import time
import torch
import traceback
from collections import defaultdict, OrderedDict
from filterSourceFilesForTraining import filterAndCopyCorpusFilesUsedToTrainBertOnMLM as filterCorpus
from simpletransformers.language_modeling import (LanguageModelingModel, LanguageModelingArgs)

# This variable holds the file extension to the training and eval files that are input to finetune our LM
global collatedInputToBertFileExtn
collatedInputToBertFileExtn = ".bertinput"

#######################################################################################################
# This class trains a pre-trained BERT (base-uncased) model with the same training objective i.e. MLM.
#######################################################################################################
class FinetuneAndEvalLanguageModelOnMLM:
    def __init__(self, corpusFolder, startDocIndex, endDocIndex, modelNameOrPath, learningRate, maxSeqLength, logLevel):
        log.debug("Initializing 'FinetuneAndEvalLanguageModelOnMLM' class instance..")
        self.corpusFolder = corpusFolder
        self.startDocIndex = startDocIndex
        self.endDocIndex = endDocIndex
        self.modelType = "bert"
        self.modelNameOrPath = modelNameOrPath
        self.allInputTrainFilesFolder = os.path.join(self.corpusFolder, "toinput")
        self.singleCorpusTrainFile = os.path.join(self.allInputTrainFilesFolder, "merged" + collatedInputToBertFileExtn + ".train")
        self.singleCorpusEvalFile = os.path.join(self.allInputTrainFilesFolder, "merged" + collatedInputToBertFileExtn + ".eval")
        self.learningRate = learningRate  # e.g., 0.00005
        self.maxSeqLength = maxSeqLength  # e.g., 512
        self.modelCacheDir = os.path.join(self.corpusFolder, "cache")
        self.modelOutputDir = os.path.join(self.corpusFolder, "finetuned_model_on_mlm")
        self.lock = threading.Lock()
        setLogLevel(logLevel)

    def __buildTrainingAndEvalTextFiles(self, saveMultipleFilesThatWereMergedIntoSingleTrainFile=True):
        #############################################################################################
        # This method builds a single train and eval files by merging the files in the corpus folder
        # in the ratio of 80:20 (train : eval). Optionally, if the flag is true, it also saves
        # the multiple files that were merged into a single training and eval files.
        #############################################################################################
        try:
            if self.corpusFolder is None or self.corpusFolder == "" or os.path.exists(self.corpusFolder) is False:
                log.error(f"Text corpus folder '{self.corpusFolder}' DOES NOT exist!")
                return None

            # Get the list of documents in the corpus folder and loop through each document to generate tokens
            docs = glob.glob(os.path.join(self.corpusFolder, "*.txt"))
            if not docs:
                log.error(f"There are NO pre-processed files in the corpus folder '{self.corpusFolder}'.")
                return None

            # Get the list of documents that needs to be processed by the processor that runs this code
            trainingFilesBuildSuccessful = False
            evalFileBuildSuccessful = False
            docs = docs[self.startDocIndex:self.endDocIndex]
            if not docs:
                log.error(
                    f"There are NO pre-processed files in the corpus folder '{self.corpusFolder}' in the document "
                    f"range [{self.startDocIndex}:{self.endDocIndex}].")
                return None

            # Loop through each file in 'docs' and collate all the lines in each file such that the length of
            #  each line is 80% of 'self.maxSeqLen' tokens per line (the balance 20% are expected to be sub-words).
            if os.path.exists(self.allInputTrainFilesFolder) is False:
                os.mkdir(self.allInputTrainFilesFolder)
            docCnt = 0  # running count of documents that were processed thus far
            for doc in docs:
                docCnt += 1
                try:
                    # Open a pre-processed file and read all the lines
                    with open(doc, "r", encoding="utf-8") as f:
                        log.debug(f"Reading document: {doc}..")
                        # Read all the lines in the file
                        lines = []
                        for line in f:
                            # Remove \n char
                            line = line.strip()
                            if line is not None and line != "":
                                lines.append(line + ". ")

                    # Build a paragraph with total words length less than 80% of "self.maxSeqLen".
                    # This paragraph will be passed to "LineByLineTextDataset" as input.
                    paragraphs = []
                    if lines and len(lines) > 0:
                        para = ""
                        for i in range(len(lines)):
                            line = lines[i].strip()
                            if len((para + line).split()) < round(self.maxSeqLength * 0.8):
                                para += line + " "
                            else:
                                if para.strip() != "":
                                    paragraphs.append(para.strip() + "\n")
                                para = ""
                        if para.strip() != "":  # add the last para that might be less than self.maxSeqLength * 0.8
                            paragraphs.append(para.strip() + "\n")

                    # Write to either train or eval corpus files
                    if paragraphs and len(paragraphs) > 0:
                        if docCnt % 5 == 0:  # 1 out of 5 documents go into eval corpus file
                            # Write to eval corpus file
                            log.debug(f"Writing document '{doc}' to single eval corpus file '{self.singleCorpusEvalFile}'..")
                            with open(self.singleCorpusEvalFile, "a+", encoding="utf-8") as f:
                                f.writelines(paragraphs)
                        else:
                            # Write to a single training corpus file by merging this file into it
                            log.debug(f"Writing document '{doc}' to single train corpus file '{self.singleCorpusTrainFile}'..")
                            with open(self.singleCorpusTrainFile, "a+", encoding="utf-8") as f:
                                f.writelines(paragraphs)

                            # Also, write to individual training corpus file
                            if saveMultipleFilesThatWereMergedIntoSingleTrainFile:
                                outFile = os.path.join(self.allInputTrainFilesFolder, os.path.split(doc)[1] + collatedInputToBertFileExtn)
                                log.debug(f"Also, writing document '{doc}' to '{outFile}'..")
                                with open(outFile, "w", encoding="utf-8") as f:
                                    f.writelines(paragraphs)
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    err = f"Error occurred while reading lines from the document {doc}. Ignoring this doc and " \
                          f"processing next doc. Error was: {str(exc_type)}; {str(exc_value)}. "
                    log.error(err)
                    continue

            # Check if the training and eval files are successfully built.
            if os.path.exists(self.singleCorpusTrainFile):
                if os.stat(self.singleCorpusTrainFile).st_size > 1024 * 1024:  # should be at least 1 MB
                    trainingFilesBuildSuccessful = True
            if os.path.exists(self.singleCorpusEvalFile):
                if os.stat(self.singleCorpusEvalFile).st_size > 1024 * 1024:  # should be at least 1 MB
                    evalFileBuildSuccessful = True

            return (trainingFilesBuildSuccessful, evalFileBuildSuccessful)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = f"Error occurred while building training and eval files from the corpus folder '{self.corpusFolder}'. Error is: {str(exc_type)}; {str(exc_value)}. "
            raise Exception(err)

    def finetunePretrainedBertOnMLM(self, saveMultipleFilesThatWereMergedIntoSingleTrainFile=True):
        ############################################################################################
        # This method finetunes a pre-trained BERT model (base-uncased) on MLM using SEC data as follows:
        #   1. Builds training and eval files.
        #   2. Uses Simple Transformers "LanguageModelingModel" class to batch process the
        #      above training and eval files.
        #   3. Evaluates the finetuned model using the above class.
        ############################################################################################
        try:
            if self.corpusFolder is None or self.corpusFolder == "" or os.path.exists(self.corpusFolder) is False:
                log.error(f"Text corpus folder '{self.corpusFolder}' DOES NOT exist!")
                return

            # Check if CUDA is available for doing training on a GPU system
            if torch.cuda.is_available() is False:
                log.error(
                    f"CUDA libs not found. A new language model can be trained from scratch only on a GPU system with "
                    f"CUDA libs!")
                return

            startTime = time.time()
            #################################################################
            # 1. Build text corpus files for train and eval datasets.
            #################################################################
            trainOk, evalOk = self.__buildTrainingAndEvalTextFiles(saveMultipleFilesThatWereMergedIntoSingleTrainFile)

            # Check if the train and eval files are built as single text corpus file
            if trainOk is False:
                log.error(f"Error building training files to finetune pre-trained BERT on MLM objective!")
                return
            if evalOk is False:
                log.error(f"Error building evaluation file to finetune pre-trained BERT on MLM obective!")
                return

            #####################################################################
            # 2. Uses Simple Transformers "LanguageModelingModel" class to train
            #####################################################################
            log.debug(f"Building config params for SimpleTransformer..")
            transformers_logger = log.getLogger("transformers")
            transformers_logger.setLevel(log.WARNING)
            modelArgs = {"reprocess_input_data": True, "overwrite_output_dir": True, "num_training_epochs": 2,
                         "dataset_type": "simple",
                         "encoding": "utf-8", "cache_dir": self.modelCacheDir, "do_lower_case": True,
                         "learning_rate": self.learningRate, "max_seq_length": self.maxSeqLength,
                         "sliding_window": True, "stride": 0.7, "handle_chinese_chars": False,}
            log.debug(f"Finished building config params for SimpleTransformer.")

            log.debug(f"Initializing SimpleTransformer's LanguageModelingModel class..")
            model = LanguageModelingModel(model_type=self.modelType, model_name=self.modelNameOrPath, args=modelArgs)
            log.debug(f"Finished initializing SimpleTransformer's LanguageModelingModel class.")

            log.info(f"Started finetuning pre-trained BERT (large-uncased) on same MLM objective with SEC data..")
            model.train_model(train_file=self.singleCorpusTrainFile, eval_file=self.singleCorpusEvalFile, output_dir=self.modelOutputDir, show_running_loss=True,)
            log.info(f"Finished finetuning and saving a pre-trained BERT (large-uncased) model on MLM with SEC data. "
                     f"Check the folder '{self.modelOutputDir}' for finetuned weights.")
            log.info(f"It took {round((time.time()-startTime)/3600, 1)} hours to finetune a pre-trained BERT model on "
                     f"MLM with SEC data from the corpus '{self.corpusFolder}'")

            # Evaluation
            log.info(f"Started evaluating the finetuned BERT (large-uncased) model on: a) perplexity, and b) eval_loss.")
            model.eval_model(eval_file=self.singleCorpusEvalFile, output_dir=self.modelOutputDir, verbose=True, silent=False)
            log.info(f"Finished evaluation of the finetuned BERT (large-uncased) model on MLM with SEC data. Check "
                     f"the evaluation results in the folder '{self.modelOutputDir}'.")
            log.info(f"It took {round((time.time()-startTime)/3600, 1)} hours to evaluate the finetuned BERT model on MLM.'")

        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = f"** ERROR ** occurred while finetuning a pre-trained BERT model and evaluating it. Error is: {exc_type}; {exc_value}."
            raise Exception(err)


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
    print(f"Setting log level to {level}")


if __name__ == "__main__":
    print("Total # of arguments passed to main() is {0}".format(len(sys.argv)))
    if len(sys.argv) < 3:
        print(
            "** ERROR ** 1) Corpus folder that has the list of pre-processed documents, and 2) whether to filter the corpus are required!")
        print(
            "Usage:\n\t<this script> <corpus folder that has pre-processed documents> <filter corpus (Y/N)> <log level (E/D/I)>")
    else:
        # Get corpus folder as given by the user in the command line
        _corpusFolder = sys.argv[1]
        if os.path.exists(_corpusFolder) is False:
            print(f"** ERROR ** Corpus folder '{_corpusFolder}' DOES NOT exist!")
            print(
                "Usage:\n\t<this script> <corpus folder that has pre-processed documents> <filter corpus (Y/N)> <log level (E/D/I)>")
        else:
            # Convert the path to absolute path
            _corpusFolder = os.path.abspath(_corpusFolder)

            # Check if the user wants to train with a reduced corpus to save training time
            _filterCorpus = False
            if not sys.argv[2] is None:
                if sys.argv[2].lower() == "y":
                    _filterCorpus = True
            if not _filterCorpus:
                    log.warning(f"You've selected to finetune a pre-trained BERT model with total corpus. It might take considerable time!")

            # Get log level set by the user in the command line
            _logLevel = "E"  # default to log.ERROR
            if len(sys.argv) == 4:
                _logLevel = sys.argv[3]  # over default log level as set by the user
            setLogLevel(_logLevel)

            # Get list of clean files (pre-processing completed) from the corpus folder
            _cleanedFiles = glob.glob(os.path.join(_corpusFolder, "*.txt"))
            if not _cleanedFiles:
                log.error(f"There are NO pre-processed (clean) documents in the corpus folder '{_corpusFolder}'!")
            else:
                _corpusLen = len(_cleanedFiles)
                log.info(f"There are a total of {_corpusLen} pre-processed (NOT filtered) files in the folder '{_corpusFolder}'.")
                try:
                    _corpusFolderFiltered = os.path.join(_corpusFolder, "filteredCorpus")
                    if _filterCorpus:
                        if not os.path.exists(_corpusFolderFiltered):
                            os.mkdir(_corpusFolderFiltered)
                        else:
                            # Delete old files if exists in the folder
                            oldFiles = glob.glob(os.path.join(_corpusFolderFiltered, "*.*"))
                            for oldFile in oldFiles:
                                os.remove(oldFile)

                        # Filter the total corpus and copy the filtered files to the new corpus filtered folder
                        filterCorpus(_corpusFolder, _corpusFolderFiltered)
                        if not os.listdir(_corpusFolderFiltered):
                            raise Exception(f"Error filtering and copying the filtered files to the folder '{_corpusFolderFiltered}. Cannot continue with training.")
                        else:
                            _corpusLen = len(os.listdir(_corpusFolderFiltered))
                            log.info(f"There are a total of {_corpusLen} pre-processed (filtered) files in the folder '{_corpusFolderFiltered}'.")

                    # Set the model name or path to "bert-base-uncased"
                    _modelPath = "bert-base-uncased"

                    # Start finetuning with different hyper-parameters as defined in the file 'hyper.params"
                    _hyperParamsFiles = glob.glob(os.path.join(_corpusFolder, "hyp*.params"))
                    _learningRate = 0.00005  # set some default value
                    _maxSeqLen = 256  # set some default value
                    if not _hyperParamsFiles is None and len(_hyperParamsFiles) > 0:
                        # Read the hyper parameters: Learning Rate and Max Sequence Length as defined in the "first" matching file
                        _lrs = []
                        _msls = []
                        with open(_hyperParamsFiles[0], "r", encoding="utf-8") as f:
                            cols = f.read().split()  # should have split the 2-column line into Learning rate and Max seq len
                            if cols:
                                _lrs.append(cols[0])
                                _msls.append(cols[1])
                        if _lrs and _msls:
                            # Start training with different hyper parameters
                            for i in range(len(_lrs)):
                                cls = FinetuneAndEvalLanguageModelOnMLM(
                                    _corpusFolderFiltered if _filterCorpus else _corpusFolder, 0, _corpusLen - 1,
                                    _modelPath, _lrs[i], _msls[i], _logLevel)
                                cls.finetunePretrainedBertOnMLM(True)  # also, generate the list of files that were used to merge into a single merged training corpus file
                                cls = None
                        else:
                            # Start training with hard-coded values as above
                            cls = FinetuneAndEvalLanguageModelOnMLM(
                                _corpusFolderFiltered if _filterCorpus else _corpusFolder, 0, _corpusLen - 1,
                                _modelPath, _learningRate, _maxSeqLen, _logLevel)
                            cls.finetunePretrainedBertOnMLM(True)
                            cls = None
                    else:
                        # Start training with hard-coded values as above
                        cls = FinetuneAndEvalLanguageModelOnMLM(
                            _corpusFolderFiltered if _filterCorpus else _corpusFolder, 0, _corpusLen - 1, _modelPath,
                            _learningRate, _maxSeqLen, _logLevel)
                        cls.finetunePretrainedBertOnMLM(True)
                        cls = None
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    err = f"\n\t {exc_type}; {exc_value}"
                    log.error(err)
