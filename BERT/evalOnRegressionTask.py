###################################################################################################
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 22:05:51 2020

@author: Srikanth Thirumalasetti (Roll #2019900090)
"""

""" This file is part of Main project on 'Learning representations in Financial domain' """

""" It evaluates a finetuned BERT model (base-uncased) on sentiment analysis/regression task. """

""" The script uses SimpleTransformers's ClassificationModel for finetuning.  """

""" We use wandb to optimize our hyper params (learning rate and # of epochs) using "accuracy" as the key metric
    to evaluate the performance and store the best model."""

""" Since, we've limited training dataset, we'll merge the posts and headlines training files 
    into a single training file and split it into train and eval datasets. """

""" The model is:
     1. first trained on the train set (self.trainDataFrame), 
     2. next, the wandb sweep is evaluated on the validation set (self.evalDataFrame).
     
     The best model as measured by the minimum mean square error corresponding to the hyper parameter values is saved to folder: self.bestModelOutputDir.
"""
###################################################################################################
import glob
import json
import logging as log
import math
import multiprocessing
import os
import pandas
import re
import sklearn
import subprocess
import sys
import threading
import time
import torch
import traceback
import wandb
from collections import defaultdict, OrderedDict
from filterSourceFilesForTraining import filterAndCopyCorpusFilesUsedToTrainBertOnMLM as filterCorpus
from nltk.tokenize import sent_tokenize
from preprocess import preprocess_seq # make sure that the pre-process.py file is in the parent folder, else, the script errors out
from sklearn.model_selection import train_test_split
from simpletransformers.classification import (ClassificationModel, ClassificationArgs)

# This variable is used as header for the train and validation (test) dataframes
global HEADER_COLS
HEADER_COLS = ["sentence", "sentiment_score"]

# This variable is the WandB API key that is used to log training and eval params in real-time to WandB server
global WAND_API_KEY, WAND_PROJECT_NAME
WAND_API_KEY = "01b06361bbf14e2d29e535b7ae84a9f3716365a4"
WAND_PROJECT_NAME = "bert-base-finetune-mlm-sec-data-regresn"


############################################################################
# This class evaluates a finetuned BERT model on sentiment analysis/regression  task.
############################################################################
class EvalLanguageModelOnRegression:
    def __init__(self, modelNameOrPath, trainFile, maxSeqLen, wandb_sweep_config, wandb_sweep_defaults, logLevel):
        log.debug("Initializing 'EvalLanguageModelOnRegression' class instance..")
        self.modelType = "bert"
        self.modelNameOrPath = modelNameOrPath
        self.trainFile = trainFile
        self.trainDataFrame = None
        self.evalDataFrame = None
        self.maxSeqLength = maxSeqLen
        self.wandbConfig = wandb_sweep_config
        self.wandbDefaults = wandb_sweep_defaults
        self.modelOutputDir = os.path.join(os.path.split(trainFile)[0], "finetuned_model_on_regression")
        self.bestModelOutputDir = os.path.join(self.modelOutputDir, "best_model")
        self.modelCacheDir = os.path.join(self.modelOutputDir, "cache")
        self.modelFinalEvalResultsFile = os.path.join(self.modelOutputDir, "model.eval.results")
        self.modelFinalEvalOutputs = os.path.join(self.modelOutputDir, "model.eval.outputs")
        self.modelFinalWrongPreds = os.path.join(self.modelOutputDir, "model.predictions.wrong.results")
        self.lock = threading.Lock()
        setLogLevel(logLevel)

    def __parseJsonFile(self, trainJsonFile):
        ##############################################################################
        # This file parses the given json file and builds a dataframe of two columns:
        #   "sentence", "sentiment_score".
        # It additionally applies pre-processing on the sentence column.
        ##############################################################################
        sentSentiscores = list(tuple())
        with open(trainJsonFile, "r", encoding="utf-8") as f:
            root = json.load(f)
            for id, idVals in root.items():
                if isinstance(idVals, dict):
                    sent = ""
                    for sentOrInfoKey, sentOrInfoVals in idVals.items():
                        if sentOrInfoKey.lower() == HEADER_COLS[0]: # sentence
                            sent = self.__preprocessSequenceWithoutBreakingSentence(sentOrInfoVals).strip()
                        elif sentOrInfoKey.lower() == "info":
                            for i, sentOrInfoVal in enumerate(sentOrInfoVals):
                                if isinstance(sentOrInfoVal, dict):
                                    for k, v in sentOrInfoVal.items():
                                        if isinstance(k, str):
                                            if k.lower() == HEADER_COLS[1]: # sentiment_score
                                                if not (sent, float(v)) in sentSentiscores:
                                                    sentSentiscores.append((sent, float(v)))
                                                    continue
        return pandas.DataFrame.from_records(sentSentiscores, columns=HEADER_COLS)

    def __preprocessSequenceWithoutBreakingSentence(self, sequence):
        ##############################################################################################
        # This method ensures that if multiple sentences are passed for pre-processing, the sequence
        # is pre-processed as individual sentence.
        #############################################################################################
        try:
            seqsPP = []
            sequences = sent_tokenize(sequence)
            if sequences:
                for seq in sequences:
                    seqsPP.append(preprocess_seq(seq))
            if seqsPP:
                return ".".join(seqsPP).strip()
            else:
                return sequence
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = f"Error occurred while pre-processing the sequence '{sequence}'. Error is: {str(exc_type)}; {str(exc_value)}."
            log.error(err)
            return sequence

    def __buildTrainingAndEvalDataFrames(self):
        #######################################################################################
        # This method builds training and eval dataframes from the given input training files.
        #######################################################################################
        try:
            log.debug(f"Started building training dataset.")
            df = self.__parseJsonFile(self.trainFile)
            log.debug(f"Finished building training dataset.")

            # Split the training dataset into train and eval in the ratio of 50:50
            self.trainDataFrame, self.evalDataFrame = train_test_split(df, test_size=0.5, random_state=4, shuffle=True)
            self.trainDataFrame.columns = HEADER_COLS
            self.evalDataFrame.columns = HEADER_COLS

            log.debug(f"Sample training data")
            log.debug(f"{self.trainDataFrame.head(5)}")
            time.sleep(10)

            log.debug(f"Sample validation data")
            log.debug(f"{self.evalDataFrame.head(5)}")
            time.sleep(10)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = f"Error occurred while building training and eval dataframes. Error is: {str(exc_type)}; {str(exc_value)}."
            raise Exception(err)

    def finetuneBertOnRegressionTask(self):
        ########################################################################################
        # This method evaluates the finetuned BERT model on sentiment analysis/regression task.
        ########################################################################################
        try:
            # Build training and eval dataframes
            self.__buildTrainingAndEvalDataFrames()

            # Check to make sure that training and eval data frames are built
            if self.trainDataFrame is None or self.evalDataFrame is None:
                log.error(f"Error building training and eval dataframes. Cannot evaluate the finetuned model on sentiment analysis/regression  task.")
                return

            # Check if CUDA is available for doing training on a GPU system
            if torch.cuda.is_available() is False:
                log.warning(f"CUDA libs not found. It is prefered to do finetuning on on a GPU system with CUDA libs!")

            # Build WandB sweep params that are used to automatically pick up the hyper-params during training
            subprocess.run(["wandb", "login", WAND_API_KEY])
            time.sleep(1)
            sweep_defaults = self.wandbDefaults
            sweep_id = wandb.sweep(self.wandbConfig, project=WAND_PROJECT_NAME)

            # Start training
            startTime = time.time()
            def train():
                wandb.init(WAND_PROJECT_NAME)
                modelArgs = { "max_seq_length": self.maxSeqLength, "output_dir": self.modelOutputDir, "overwrite_output_dir": True, "best_model_dir": self.bestModelOutputDir,
                              "wandb_project": WAND_PROJECT_NAME, "num_training_epochs": wandb.config.epochs, "learning_rate": wandb.config.learning_rate,
                              "do_lower_case": True, "cache_dir": self.modelCacheDir, "encoding": "utf-8", "train_batch_size": 5, "eval_batch_size": 5,
                              "evaluate_during_training_steps": 50, "evaluate_during_training_verbose": True, "logging_steps": 5, "sliding_window": True,
                              "reprocess_input_data": True, "evaluate_during_training": True, "use_multiprocessing": False, "regression": True }
                model = ClassificationModel(self.modelType, self.modelNameOrPath, args=modelArgs, sweep_config=wandb.config, use_cuda=torch.cuda.is_available(), num_labels=1)

                # Training
                try:
                    log.info(f"Started finetuning BERT on sentiment analysis/regression task..")
                    model.train_model(train_df=self.trainDataFrame, eval_df=self.evalDataFrame, show_running_loss=True, output_dir=self.modelOutputDir,
                                      mse=sklearn.metrics.mean_squared_error, r2Score=sklearn.metrics.r2_score,)
                    log.info(f"Finished training and evaluation of our finetuned model on sentiment analysis/regression task. Check the folder '{self.modelOutputDir}' for finetuned weights.")
                    log.info(f"It took {round((time.time() - startTime) / 3600, 1)} hours to train/finetune BERT model on sentiment analysis/regression task.")
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    err = f"Error occurred while training finetuned model on sentiment analysis/regression task. Error is: {str(exc_type)}; {str(exc_value)}."
                    log.error(err)

                wandb.join()

            wandb.agent(sweep_id, function=train)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = f"** ERROR ** occurred while finetuning a pre-trained BERT model and evaluating it on sentiment analysis/regression task. Error is: {exc_type}; {exc_value}."
            raise Exception(err)


def mergeJsonFiles(file1, file2, outFile):
    ##############################################################################
    # This file merges the two given json files with the same structure into the
    # given outfile. It is important that schema is same for the two json files.
    ##############################################################################
    with open(file1, "r", encoding="utf-8") as f1, open(file2, "r", encoding="utf-8") as f2:
        dict1 = json.load(f1)
        dict2 = json.load(f2)
    for k, v in dict2.items():
        if not k in dict1.keys():
            dict1[k] = v
    with open(outFile, "w", encoding="utf-8") as f:
        json.dump(dict1, f)


def setLogLevel(level):
    ##########################################################
    # This method sets the log level for the default logger.
    ##########################################################
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
        print("** ERROR ** 1) Finetuned model directory, 2) Training file1 and 3) Training file2 are required!")
        print("Usage:\n\t<this script> <Finetuned model directory> <Training file1 with path> <Training file2 with path> <log level (E/D/I)>")
    else:
        # Get finetuned model folder as given by the user in the command line
        _finetunedModelFolder = sys.argv[1] # set this variable to value: "bert-base-uncased" to evaluate using Huggingface's pre-trained BERT embeddings
        if os.path.exists(_finetunedModelFolder) is False:
            print(f"** ERROR ** Finetuned model folder '{_finetunedModelFolder}' DOES NOT exist!")
            print("Usage:\n\t<this script> <Finetuned model directory> <Training file1 with path> <Training file2 with path> <log level (E/D/I)>")
        else:
            # Convert the path to absolute path
            _finetunedModelFolder = os.path.abspath(_finetunedModelFolder)

            # Get the training files
            _files = [sys.argv[2], sys.argv[3]]
            _notExists = False
            for _file in _files:
                if os.path.exists(_file) is False:
                    print(f"** ERROR ** Finetuned model folder '{_finetunedModelFolder}' DOES NOT exist!")
                    print("Usage:\n\t<this script> <Finetuned model directory> <Training file1 with path> <Training file2 with path> <log level (E/D/I)>")
                    _notExists = True
                    continue

            _trainFile = None
            if _notExists is False:
                _trainFile = os.path.join(os.path.split(os.path.abspath(_files[0]))[0], "merged_train.json")
                # Merge training files
                mergeJsonFiles(os.path.abspath(_files[0]), os.path.abspath(_files[1]), _trainFile)

            # Get log level set by the user in the command line
            _logLevel = "E"  # default to log.ERROR
            if len(sys.argv) == 7:
                _logLevel = sys.argv[6]  # over default log level as set by the user
            setLogLevel(_logLevel)

            # Evaluate the finetuned model on sentiment analysis/regression  task
            try:
                # Start finetuning with different hyper-parameters
                _maxSeqLen = 192 # setting to same sequence value that was used for finetuning the model on MLM objective
                _learningRates = [5e-5, 2e-5]  # set diff LRs
                _epochs = [3, 5, 10]  # set total epochs that we'd like to run
                _wandb_sweep_defaults = {'learning_rate': _learningRates[0], 'epochs': _epochs[0]} # set some default values
                _wandb_sweep_config = {'method': 'grid', "metric": {"name": "mse", "goal": "minimize"},
                                       'parameters': {'learning_rate': {'values': _learningRates}, 'epochs': {'values': _epochs}},
                                       "early_terminate": {"type": "hyperband", "min_iter": 5, }, }

                # Initialize training class and start training with hyper-parms configured in wandb
                if not _trainFile is None:
                    cls = EvalLanguageModelOnRegression(_finetunedModelFolder, _trainFile, _maxSeqLen, _wandb_sweep_config, _wandb_sweep_defaults, _logLevel)
                    cls.finetuneBertOnRegressionTask()
                    cls = None
                else:
                    raise Exception("Train file is None!")
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                err = f"\n\t {exc_type}; {exc_value}"
                log.error(err)
