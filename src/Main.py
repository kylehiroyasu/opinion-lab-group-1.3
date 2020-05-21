import os
from pathlib import Path
import sys

import numpy as np
import torch as t
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, BertEmbeddings

import preprocess
from Trainer import Trainer
from Dataset import AspectDataset, dfToDataset, collate
from Model import Model
from Learners import Learner_Classification, Learner_Clustering
from Loss import KCL, MCL, Class2Simi

ROOT = '/home/ibes222/Documents/Master/NLPLab/GitHub'
DATA = ROOT + '/data'
SRC =  ROOT + '/src'
RAW_DATA = DATA + '/raw'
RAW_FILES = [
    'ABSA16_Laptops_Train_SB1.xml',
    'ABSA16_Laptops_Test_SB1_GOLD.xml',
    'ABSA16_Restaurants_Train_SB1.xml',
    'ABSA16_Restaurants_Test_SB1_GOLD.xml'
]
print(ROOT)

laptops_train = preprocess.load_data_as_df(RAW_DATA + "/" + RAW_FILES[0])
laptops_test = preprocess.load_data_as_df(RAW_DATA + "/" + RAW_FILES[1])

restaurants_train = preprocess.load_data_as_df(RAW_DATA + "/" + RAW_FILES[2])
restaurants_test = preprocess.load_data_as_df(RAW_DATA + "/" + RAW_FILES[3])

laptop_entities = {"BATTERY": 0, "COMPANY": 1, "CPU": 2, "DISPLAY": 3, "FANS_COOLING": 4, "GRAPHICS": 5, "HARDWARE": 6, "HARD_DISC": 7, "KEYBOARD": 8, "LAPTOP": 9, "MEMORY": 10, "MOTHERBOARD": 11, "MOUSE": 12, "MULTIMEDIA_DEVICES": 13, "OPTICAL_DRIVES": 14, "OS": 15, "PORTS": 16, "POWER_SUPPLY": 17, "SHIPPING": 18, "SOFTWARE": 19, "SUPPORT": 20, "WARRANTY": 21, "NaN": 22}
laptop_attributes = {"CONNECTIVITY": 0, "DESIGN_FEATURES": 1, "GENERAL": 2, "MISCELLANEOUS": 3, "OPERATION_PERFORMANCE": 4,"PORTABILITY": 5, "PRICE": 6, "QUALITY": 7, "USABILITY": 8, "NaN": 9}
restaurant_entities = {"AMBIENCE": 0, "DRINKS": 1, "FOOD": 2, "LOCATION": 3, "RESTAURANT": 4, "SERVICE": 5, "NaN": 6}
restaurant_attributes = {"GENERAL": 0, "MISCELLANEOUS": 1, "PRICES": 2, "QUALITY": 3, "STYLE_OPTIONS": 4, "NaN": 5}

glove_embeddings = WordEmbeddings('glove')
hidden_dim = 100
output_dim = len(restaurant_entities)

train_dataset = dfToDataset(restaurants_train, restaurant_entities, restaurant_attributes, glove_embeddings)
test_dataset = dfToDataset(restaurants_test, restaurant_entities, restaurant_attributes, glove_embeddings)

print("Loaded dataset")

param = {
    "embedding_dim": hidden_dim,
    "output_dim": output_dim,
    "epochs": 40,
    "lr": 0.01,
    "batch_size": 256,
    "use_padding": False,
    "validation_percentage": 0.1,
    "binary_sampling_percentage": 0.5,
    "cuda": False,
    "use_kcl": True,
    "with_supervised": False,
    "use_micro_average": True,
    "train_entities": True
}

trainer = Trainer(train_dataset, param)
model = trainer.train()
model = trainer.train_classifier()