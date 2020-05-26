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
from Dataset import AspectDataset, dfToBinarySamplingDatasets, collate_padding
from Model import Model
from Learners import Learner_Clustering
from Loss import KCL, MCL, Class2Simi

# Because we are in the source directory, we have to go one directory upwards
ROOT = os.getcwd()
DATA = os.path.join(ROOT,'data')
SRC =  os.path.join(ROOT, 'src')
RAW_DATA = os.path.join(DATA, 'raw')
RAW_FILES = [
    'ABSA16_Laptops_Train_SB1.xml',
    'ABSA16_Laptops_Test_SB1_GOLD.xml',
    'ABSA16_Restaurants_Train_SB1.xml',
    'ABSA16_Restaurants_Test_SB1_GOLD.xml'
]
print(ROOT)

laptops_train = preprocess.load_data_as_df(os.path.join(RAW_DATA, RAW_FILES[0]))
laptops_test = preprocess.load_data_as_df(os.path.join(RAW_DATA, RAW_FILES[1]))

restaurants_train = preprocess.load_data_as_df(os.path.join(RAW_DATA, RAW_FILES[2]))
restaurants_test = preprocess.load_data_as_df(os.path.join(RAW_DATA, RAW_FILES[3]))

train_attributes = True
train_restaurant = True
target_class = "QUALITY"

laptop_entities = {"BATTERY": 0, "COMPANY": 1, "CPU": 2, "DISPLAY": 3, "FANS_COOLING": 4, "GRAPHICS": 5, "HARDWARE": 6, "HARD_DISC": 7, "KEYBOARD": 8, "LAPTOP": 9, "MEMORY": 10, "MOTHERBOARD": 11, "MOUSE": 12, "MULTIMEDIA_DEVICES": 13, "OPTICAL_DRIVES": 14, "OS": 15, "PORTS": 16, "POWER_SUPPLY": 17, "SHIPPING": 18, "SOFTWARE": 19, "SUPPORT": 20, "WARRANTY": 21, "NaN": 22}
laptop_attributes = {"CONNECTIVITY": 0, "DESIGN_FEATURES": 1, "GENERAL": 2, "MISCELLANEOUS": 3, "OPERATION_PERFORMANCE": 4,"PORTABILITY": 5, "PRICE": 6, "QUALITY": 7, "USABILITY": 8, "NaN": 9}
restaurant_entities = {"AMBIENCE": 0, "DRINKS": 1, "FOOD": 2, "LOCATION": 3, "RESTAURANT": 4, "SERVICE": 5, "NaN": 6}
restaurant_attributes = {"GENERAL": 0, "MISCELLANEOUS": 1, "PRICES": 2, "QUALITY": 3, "STYLE_OPTIONS": 4, "NaN": 5}

if train_restaurant:
    train_set = restaurants_train
    test_set = restaurants_test
    entities = restaurant_entities
    attributes = restaurant_attributes
else:
    train_set = laptops_train
    test_set = laptops_test
    entities = laptops_entities
    attributes = laptops_attributes
    
embeddings = WordEmbeddings('glove')
hidden_dim = 100
# This is the dimension of the output of the ABAE model, the classification model gets this as input
# It does not need to be related to the number of classes etc.
output_dim = len(attributes if train_attributes else entities)

train_dataset, other_train_dataset = dfToBinarySamplingDatasets(train_set, train_attributes, target_class, embeddings)
test_dataset, other_test_dataset = dfToBinarySamplingDatasets(test_set, train_attributes, target_class, embeddings)

print("Loaded dataset")

param = {
    "embedding_dim": hidden_dim,
    "output_dim": output_dim,
    "classification_dim": 1,
    "epochs": 500,
    "lr": 5e-4,
    "lr_decay_epochs": 350,
    "batch_size": 512,
    "validation_percentage": 0.1,
    "binary_sampling_percentage": 1,
    "cuda": True,
    "use_kcl": False,
    "use_micro_average": True,
    "train_entities": True,
    "target_class": target_class,
    "freeze": True,
    "save_training_records": True,
    "records_data_path": 'records/'+ ('restaurants/' if train_restaurant else 'laptop/') + ('attribute/' if train_attributes else 'entity/')
}

trainer = Trainer(train_dataset, param, other_train_dataset)
model = trainer.train()
param["lr"] = 0.0005
param["batch_size"] = 52
model = trainer.train_classifier(freeze=param["freeze"], new_param=param)