###################################################################################################
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 22:05:51 2020

@author: Srikanth Thirumalasetti (Roll #2019900090)
"""

""" This file is part of Main project on 'Learning representations in Financial domain' """

""" It evaluates a finetuned BERT model (base-uncased) on binary classification task. """

""" The script uses SimpleTransformers's ClassificationModel for finetuning.  """

""" We use wandb to optimize our hyper params (learning rate and # of epochs) using "accuracy" as the key metric
    to evaluate the performance and store the best model."""

""" The model is:
     1. first trained on the train set (self.trainDataFrame), 
     2. next, the wandb sweep is evaluated on the validation set (self.evalDataFrame)
     
     The best model as measured by the maximum accuracy corresponding to the hyper parameter values is saved to folder: self.bestModelOutputDir.
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
from simpletransformers.classification import (ClassificationModel, ClassificationArgs)

# This variable is used as header for the train and validation dataframes
global HEADER_COLS
HEADER_COLS = ["tweet", "relation"]

# This variable is the WandB API key that is used to log training and eval params in real-time to WandB server
global WAND_API_KEY, WAND_PROJECT_NAME
WAND_API_KEY = "01b06361bbf14e2d29e535b7ae84a9f3716365a4"
WAND_PROJECT_NAME = "bert-base-finetune-mlm-sec-data-binary-cls"


############################################################################
# This class evaluates a finetuned BERT model on binary classification task.
############################################################################
class EvalLanguageModelOnBinCls:
    def __init__(self, modelNameOrPath, trainFile, evalFile, maxSeqLen, wandb_sweep_config, wandb_sweep_defaults, logLevel):
        log.debug("Initializing 'EvalLanguageModelOnBinCls' class instance..")
        self.modelType = "bert"
        self.modelNameOrPath = modelNameOrPath
        self.trainFile = trainFile
        self.evalFile = evalFile
        self.trainDataFrame = None
        self.evalDataFrame = None
        self.maxSeqLength = maxSeqLen
        self.wandbConfig = wandb_sweep_config
        self.wandbDefaults = wandb_sweep_defaults
        self.modelOutputDir = os.path.join(os.path.split(trainFile)[0], "finetuned_model_on_bin_cls")
        self.bestModelOutputDir = os.path.join(self.modelOutputDir, "best_model")
        self.modelCacheDir = os.path.join(self.modelOutputDir, "cache")
        self.modelFinalEvalResultsFile = os.path.join(self.modelOutputDir, "model.eval.results")
        self.modelFinalEvalOutputs = os.path.join(self.modelOutputDir, "model.eval.outputs")
        self.modelFinalWrongPreds = os.path.join(self.modelOutputDir, "model.predictions.wrong.results")
        self.lock = threading.Lock()
        setLogLevel(logLevel)

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
        ##############################################################################################
        # This method builds training and eval dataframes from the given input training and dev files.
        #############################################################################################
        try:
            tweetsDictTrain = {}
            tweetsDictEval = {}
            with open(self.trainFile, "r", encoding="utf-8") as fR:
                tweetsDictTrain = json.load(fR)
            log.debug(f"Finished building training file.")
            with open(self.evalFile, "r", encoding="utf-8") as fD:
                tweetsDictEval = json.load(fD)
            log.debug(f"Finished building eval file.")

            # Columns are: 'tweet', 'target_num', 'offset', 'target_cashtag' and 'relation'
            # Our classification task is: Given a tweet, tell whether the relation is 0 or 1
            if tweetsDictTrain:
                log.debug(f"Started generating pandas dataframe for training..")
                df = pandas.DataFrame.from_dict(tweetsDictTrain)
                self.trainDataFrame = df.iloc[0:len(df.index), [0,4]]
                self.trainDataFrame.columns = HEADER_COLS
                self.trainDataFrame[HEADER_COLS[0]] = self.trainDataFrame[HEADER_COLS[0]].map(lambda sent: self.__preprocessSequenceWithoutBreakingSentence(sent))

            if tweetsDictEval:
                log.debug(f"Started generating pandas dataframe for evaluation..")
                df = pandas.DataFrame.from_dict(tweetsDictEval)
                self.evalDataFrame = df.iloc[0:len(df.index), [0,4]]
                self.evalDataFrame.columns = HEADER_COLS
                self.evalDataFrame[HEADER_COLS[0]] = self.evalDataFrame[HEADER_COLS[0]].map(lambda sent: self.__preprocessSequenceWithoutBreakingSentence(sent))
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = f"Error occurred while building training and eval dataframes. Error is: {str(exc_type)}; {str(exc_value)}."
            raise Exception(err)

    def finetuneBertOnBinClsTask(self):
        ###############################################################################
        # This method evaluates the finetuned BERT model on binary classification task.
        ###############################################################################
        try:
            # Build training and eval dataframes
            self.__buildTrainingAndEvalDataFrames()

            # Check to make sure that training and eval data frames are built
            if self.trainDataFrame is None or self.evalDataFrame is None:
                log.error(f"Error building training and eval dataframes. Cannot evaluate the finetuned model on binary classification task.")
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
                              "reprocess_input_data": True, "evaluate_during_training": True, "use_multiprocessing": True }

                model = ClassificationModel(self.modelType, self.modelNameOrPath, args=modelArgs, sweep_config=wandb.config, use_cuda=torch.cuda.is_available(),)

                # Training and evaluation
                try:
                    log.info(f"Started training/finetuning BERT on binary classification task..")
                    model.train_model(train_df=self.trainDataFrame, eval_df=self.evalDataFrame, show_running_loss=True,
                                      output_dir=self.modelOutputDir,
                                      mcc=sklearn.metrics.matthews_corrcoef,
                                      f1=sklearn.metrics.f1_score,
                                      acc=sklearn.metrics.accuracy_score,
                                      recall_score=sklearn.metrics.recall_score, )
                    log.info(f"Finished finetuning and evaluating our fine-tuned model on binary classification task. Check the folder '{self.modelOutputDir}' for finetuned weights.")
                    log.info(f"It took {round((time.time() - startTime) / 3600, 1)} hours to finetune and evaluate ou fine-tuned model on binary classification task.")
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    err = f"Error occurred while training and evaluating the finetuned model on binary classification task. Error is: {exc_type}; {exc_value}."
                    log.error(err)

                wandb.join()

            wandb.agent(sweep_id, function=train)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = f"** ERROR ** occurred while finetuning a pre-trained BERT model and evaluating it on binary classification task. Error is: {exc_type}; {exc_value}."
            log.error(err)


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
        print("** ERROR ** 1) Finetuned model directory, 2) Training file, and 3) Dev file are required!")
        print("Usage:\n\t<this script> <Finetuned model directory> <Training file with path> <Dev file with path> <log level (E/D/I)>")
    else:
        # Get finetuned model folder as given by the user in the command line
        _finetunedModelFolder = sys.argv[1] # set this variable to value: "bert-base-uncased" to evaluate using Huggingface's pre-trained BERT embeddings
        if os.path.exists(_finetunedModelFolder) is False:
            print(f"** ERROR ** Finetuned model folder '{_finetunedModelFolder}' DOES NOT exist!")
            print("Usage:\n\t<this script> <Finetuned model directory> <Training file with path> <Dev file with path> <log level (E/D/I)>")
        else:
            # Convert the path to absolute path
            _finetunedModelFolder = os.path.abspath(_finetunedModelFolder)

            # Get the training and dev files
            _trainFile = sys.argv[2]
            _trainFile = os.path.abspath(_trainFile)
            _devFile = sys.argv[3]
            _devFile = os.path.abspath(_devFile)

            # Get log level set by the user in the command line
            _logLevel = "E"  # default to log.ERROR
            if len(sys.argv) == 5:
                _logLevel = sys.argv[4]  # over default log level as set by the user
            setLogLevel(_logLevel)

            # Evaluate the finetuned model on binary classification task
            try:
                # Start finetuning with different hyper-parameters
                _maxSeqLen = 192 # setting to same sequence value that was used for finetuning the model on MLM objective
                _learningRates = [5e-5, 2e-5, 1e-5]  # set three diff LRs
                _epochs = [3, 5, 10]  # set total epochs that we'd like to run
                _wandb_sweep_defaults = {'learning_rate': _learningRates[0], 'epochs': _epochs[0]} # set some default values
                _wandb_sweep_config = {'method': 'grid', "metric": {"name": "mcc", "goal": "maximize"},
                                       'parameters': {'learning_rate': {'values': _learningRates}, 'epochs': {'values': _epochs}},
                                       "early_terminate": {"type": "hyperband", "min_iter": 5, },}

                # Initialize training class and start training
                cls = EvalLanguageModelOnBinCls(_finetunedModelFolder, _trainFile, _devFile, _maxSeqLen, _wandb_sweep_config, _wandb_sweep_defaults, _logLevel)
                cls.finetuneBertOnBinClsTask()
                cls = None
            except:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                err = f"\n\t {exc_type}; {exc_value}"
                log.error(err)
