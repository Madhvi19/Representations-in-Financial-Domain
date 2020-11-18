###################################################################################################
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 22:05:51 2020

@author: Srikanth Thirumalasetti (Roll #2019900090)
"""

""" This file is part of Main project on 'Learning representations in Financial domain' """
""" It evaluates our finetuned BERT (base-uncased) model on the multi-class classification SEC dataset """
""" Multi-class classification is done as follows:
    A) Using SimpleTransformers "Classification" model configuration to randomly initialize a sequence classification head
        on top of our BERT finetuned model's encoder with an output size of 13 - each mapping to one of the "sector" labels 
        in the SEC summary data "df_10k_1900_org.pkl" file. 
        Data cleaning done on the above file:
            - Removed rows corresponding to the sectors with the following labels: 
                "cosmetics & fragrance inc", "herman inc", "ruger & company", "inc", "inc.", "inc (formerly acxiom)", "inc. (staten island", "incorporated", "ltd." and "na"
            - Removed rows corresponding to the value: "No business text found" in the column "text".
        After cleaning the data, finetuning on multi-class classification was done with the following 13 labels:
            1. communication services
            2. consumer discretionary
            3. consumer staples
            4. customer discretionary
            5. energy
            6. financials
            7. health care
            8. industrials
            9. information technolgy
            10. materials
            11. real estate
            12. telecommunication services
            13. utilities
    B) The summary dataset is split into train and eval datasets
    C) The model is finetuned with train dataset and evalauted on eval dataset. 
"""
###################################################################################################
import glob
import logging as log
import math
import multiprocessing as mproc
import os
import pandas
import pickle
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
from enum import Enum
from nltk.tokenize import sent_tokenize
from preprocess import preprocess_seq # make sure that the pre-process.py file is in the parent folder, else, the script errors out
from sklearn.model_selection import train_test_split as tts
from simpletransformers.classification import (ClassificationModel, ClassificationArgs)

# These variables are used in multi-processing
NUM_CPUs = mproc.cpu_count()
MULTIPLY_FACTOR = 1

# This variable are the Column labels used for training
global SECTOR_LABELS
SECTOR_LABELS = ["communication services", "consumer discretionary", "consumer staples", "customer discretionary", "energy", "financials",
                      "health care", "industrials", "information technology", "materials", "real estate", "telecommunication services", "utilities"]

# This variable is used as header for the train and validation dataframes
global HEADER_COLS
HEADER_COLS = ["text", "sector"]

global EMPTY_BUSINESS_ITEM_MSG
EMPTY_BUSINESS_ITEM_MSG = "No business text found"

# This variable is the WandB API key that is used to log training and eval params in real-time to WandB server
global WAND_API_KEY, WAND_PROJECT_NAME
WAND_API_KEY = "01b06361bbf14e2d29e535b7ae84a9f3716365a4"
WAND_PROJECT_NAME = "bert-base-finetune-mlm-sec-data-multiclass-cls"

###################################################################################################################
# This class evaluates our finetuned BERT embeddings on multi-class classification task.
#   1. Splitting the pickled data file into: train and eval datasets.
#   2. Using SimpleTransformers, finetunes a BertForSequenceClassification model on the finetuned BERT embeddings.
###################################################################################################################
class EvalLanguageModelOnMultiClassCls:
    def __init__(self, modelNameOrPath, trainFile, maxSeqLength, wandbConfig, wandbDefaults, logLevel):
        log.debug("Initializing 'EvaluateBertEmbeddingsOnMultiClassificationTask' class instance..")
        self.modelType = "bert"
        self.modelNameOrPath = modelNameOrPath
        self.trainFile = trainFile
        self.trainDataset = None
        self.evalDataset = None
        self.maxSeqLength = maxSeqLength # e.g., 512
        self.wandbConfig = wandbConfig
        self.wandbDefaults = wandbDefaults
        self.modelOutputDir = os.path.join(os.path.split(trainFile)[0], "finetuned_model_on_multi_cls")
        self.bestModelOutputDir = os.path.join(self.modelOutputDir, "best_model")
        self.modelCacheDir = os.path.join(self.modelOutputDir, "cache3")
        self.modelFinalEvalResultsFile = os.path.join(self.modelOutputDir, "model.eval.results")
        self.modelFinalEvalOutputs = os.path.join(self.modelOutputDir, "model.eval.outputs")
        self.modelFinalWrongPreds = os.path.join(self.modelOutputDir, "model.predictions.wrong.results")
        self.lock = threading.Lock()
        setLogLevel(logLevel)

    def __buildTrainingAndEvalDatasets(self):
        ###################################################################################
        # This method builds the train and eval datasets from the given pickled data file.
        ###################################################################################
        global HEADER_COLS, EMPTY_BUSINESS_ITEM_MSG
        try:
            # Check if the train file exists
            if os.path.exists(self.trainFile) is False:
                log.error(f"Pickle file '{self.trainFile}' does not exist!")
                return False

            # Check if the file has been read successfully
            dfPckl = pandas.read_pickle(self.trainFile)
            if not dfPckl is None:
                totalInitRows = len(dfPckl)
                log.debug(dfPckl[HEADER_COLS].head(50))
                time.sleep(60)

                # Remove rows corresponding to the sectors with the following labels:
                #    "cosmetics & fragrance inc", "herman inc", "ruger & company", "inc", "inc.",
                #    "inc (formerly acxiom)", "inc. (staten island", "incorporated", "ltd." and "na".
                log.debug(f"Removing rows where '{HEADER_COLS[1]} != {', '.join(SECTOR_LABELS)}'")
                dfPckl = dfPckl[dfPckl.eval(HEADER_COLS[1]).isin(SECTOR_LABELS)]
                log.debug(dfPckl[HEADER_COLS].head(50))
                time.sleep(60)

                # Remove rows corresponding to the value: "No business text found" in the column "text".
                log.debug(f"Removing rows where '{HEADER_COLS[0]} == {EMPTY_BUSINESS_ITEM_MSG}'..")
                dfPckl = dfPckl[dfPckl.eval(HEADER_COLS[0]).str.lower() != EMPTY_BUSINESS_ITEM_MSG.lower()]
                log.debug(dfPckl[HEADER_COLS].head(50))
                time.sleep(60)

                log.debug(f"Total records in the dataframe: {totalInitRows}.")
                log.debug(f"Total records in the dataframe that were removed: {totalInitRows - len(dfPckl)}.")
                log.debug(f"Total records in the dataframe for training and evaluation: {len(dfPckl)}.")

                # Apply pre-processing on the "text" column on multiple processors
                log.debug(f"Applying pre-processing on the '{HEADER_COLS[0]}' column..")
                with mproc.Pool(NUM_CPUs) as p:
                    dfPckl[HEADER_COLS[0]] = p.map(preprocessSequenceWithoutBreakingSentence, [text for text in dfPckl[HEADER_COLS[0]]])
                log.debug(dfPckl[HEADER_COLS].head(50))
                time.sleep(60)

                # Save the pre-processed dataframe to a pickle file
                try:
                    preProcPckl = os.path.join(os.path.split(self.trainFile)[0], os.path.split(self.trainFile)[1].split(".")[0] + ".preproc.pkl")
                    dfPckl.to_pickle(preProcPckl)
                    log.info(f"Successfully saved the pre-processed training file to '{preProcPckl}'.")
                except:
                    log.error("Error saving the pre-processed dataset to file.")

                # Split into train and eval datasets
                self.trainDataset, self.evalDataset = tts(dfPckl, test_size=0.33, shuffle=True, random_state=42)
                if not self.trainDataset is None and not self.evalDataset is None:
                    log.debug(f"Total records in the dataframe for training are '{len(self.trainDataset)}'.")
                    log.debug(f"Total records in the dataframe for evaluation are '{len(self.evalDataset)}'.")
                    log.info(f"Successfully generated train and eval datasets.")
                    return True
                else:
                    log.error(f"Error generating train and eval datasets. Cannot continue with finetuning.")
                    return False
            else:
                log.error(f"Error reading the pickle file '{self.trainFile}'. Cannot continue with finetuning.")
                return False
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = f"** ERROR ** Error occurred while building training and eval datasets from the pickled file '{self.trainFile}'. Error is: {str(exc_type)}; {str(exc_value)}."
            raise Exception(err)

    def finetuneBertOnMultiClassClsTask(self):
        #####################################################################################
        # This method evaluates the finetuned BERT model on multi-class classification task.
        #####################################################################################
        global SECTOR_LABELS, WAND_PROJECT_NAME, WAND_API_KEY
        try:
            # Build training and eval datasets
            if self.__buildTrainingAndEvalDatasets() is False:
                log.error(f"Error building training / eval dataset to train / eval finetuned BERT embeddings on multi-classification task! Cannot continue with evaluation.")
                return

            time.sleep(60)

            # Check if CUDA is available for doing training on a GPU system
            if torch.cuda.is_available() is False:
                log.error(f"CUDA libs not found. A new language model can be trained from scratch only on a GPU system with CUDA libs!")

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
                              "reprocess_input_data": True, "evaluate_during_training": True, "use_multiprocessing": True,
                              "labels_list": SECTOR_LABELS }

                model = ClassificationModel(self.modelType, self.modelNameOrPath, args=modelArgs, sweep_config=wandb.config, use_cuda=torch.cuda.is_available(), num_labels=len(SECTOR_LABELS), )

                # Training and evaluation
                try:
                    log.info(f"Started training/finetuning BERT on multi-class classification task..")
                    model.train_model(train_df=self.trainDataset, eval_df=self.evalDataset, show_running_loss=True,
                                      output_dir=self.modelOutputDir,
                                      mcc=sklearn.metrics.matthews_corrcoef,
                                      acc=sklearn.metrics.balanced_accuracy_score, )
                    log.info(f"Finished finetuning and evaluating our fine-tuned model on multi-class classification task. Check the folder '{self.modelOutputDir}' for finetuned weights.")
                    log.info(f"It took {round((time.time() - startTime) / 3600, 1)} hours to finetune and evaluate our fine-tuned model on multi-class classification task.")
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    err = f"Error occurred while training and evaluating the finetuned model on multi-class classification task. Error is: {exc_type}; {exc_value}."
                    log.error(err)

                wandb.join()

            wandb.agent(sweep_id, function=train)
        except:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err = f"** ERROR ** occurred while finetuning our BERT model on multi-classification task and evaluating it. Error is: {exc_type}; {exc_value}."
            raise Exception(err)


def preprocessSequenceWithoutBreakingSentence(sequence):
    ##############################################################################################
    # This method ensures that if multiple sentences are passed for pre-processing, the sequence
    # is pre-processed as individual sentence.
    #############################################################################################
    try:
        # Hack to address the issue with the script 'get_business_text.py' that is returning text with multiple \n\n
        # Remove everything between \n\n and \n\n as these are headers for a section
        sequence = sequence.replace("\n\n", "<<>>")
        sequence = re.sub(r"<<>>[^<]+<<>>", " ", sequence)
        sequence = sequence.replace("\n", " ").replace("<<>>","").strip()

        seqsPP = []
        sequences = sent_tokenize(sequence)
        if sequences:
            for seq in sequences:
                seqPP = preprocess_seq(seq.strip())
                if not seqPP is None and seqPP.strip() != "" and len(seqPP.split(" ")) > 1: # do not include empty and single word strings
                    seqsPP.append(seqPP)
        if seqsPP:
            return ". ".join(seqsPP).strip()
        else:
            return sequence
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        err = f"Error occurred while pre-processing the sequence '{sequence}'. Error is: {str(exc_type)}; {str(exc_value)}."
        log.error(err)
        return sequence

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
    print("Total # of arguments passed to evaluate BERT embeddings is {0}".format(len(sys.argv)))
    _usageMsg = "Usage:\n<this script> <Directory of trained model that has pytorch .bin model file> <Pickle file (with business text) for training> <log level (E/D/I)>"
    if len(sys.argv) < 3:
        print(_usageMsg)
    else:
        # Get finetuned model folder path
        _finetunedModelFolder = sys.argv[1]
        if os.path.exists(_finetunedModelFolder) is False:
            print(f"** ERROR ** Folder that has pytorch .bin model file '{_finetunedModelFolder}' DOES NOT exist!")
            print(_usageMsg)
        else:
            # Convert the path to absolute path
            _modelPath = os.path.abspath(_finetunedModelFolder)

            # Get log level set by the user in the command line
            _logLevel = "E"  # default to log.ERROR
            if len(sys.argv) == 4:
                _logLevel = sys.argv[3]  # over default log level as set by the user
            setLogLevel(_logLevel)

            # Get pickle file for training
            _trainFile = sys.argv[2]
            if os.path.exists(_trainFile) is False:
                print(f"** ERROR ** Pickle file '{_trainFile}' DOES NOT exist!")
                print(_usageMsg)
            else:
                try:
                    # Start finetuning with different hyper-parameters
                    _maxSeqLen = 192  # setting to same sequence value that was used for finetuning the model on MLM objective
                    _learningRates = [5e-5, 3e-5, 1e-5]  # set three diff LRs
                    _epochs = [3, 5, 8]  # set total epochs that we'd like to run
                    _wandb_sweep_defaults = {'learning_rate': _learningRates[0],
                                             'epochs': _epochs[0]}  # set some default values
                    _wandb_sweep_config = {'method': 'grid', "metric": {"name": "eval_loss", "goal": "minimize"},
                                           'parameters': {'learning_rate': {'values': _learningRates},
                                                          'epochs': {'values': _epochs}},
                                           "early_terminate": {"type": "hyperband", "min_iter": 5, }, }

                    # Initialize training class and start training
                    cls = EvalLanguageModelOnMultiClassCls(_finetunedModelFolder, _trainFile, _maxSeqLen, _wandb_sweep_config, _wandb_sweep_defaults, _logLevel)
                    cls.finetuneBertOnMultiClassClsTask()
                    cls = None
                except:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    err = f"\n\t {exc_type}; {exc_value}"
                    log.error(err)
