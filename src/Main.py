import argparse
import os
import plac
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
from Dataset import AspectDataset, dfToDataset, dfToBinarySamplingDatasets, collate
from Model import Model
from Learners import Learner_Classification, Learner_Clustering
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


LAPTOP_ENTITIES = {"BATTERY": 0, "COMPANY": 1, "CPU": 2, "DISPLAY": 3, "FANS_COOLING": 4, "GRAPHICS": 5, "HARDWARE": 6, "HARD_DISC": 7, "KEYBOARD": 8, "LAPTOP": 9, "MEMORY": 10, "MOTHERBOARD": 11, "MOUSE": 12, "MULTIMEDIA_DEVICES": 13, "OPTICAL_DRIVES": 14, "OS": 15, "PORTS": 16, "POWER_SUPPLY": 17, "SHIPPING": 18, "SOFTWARE": 19, "SUPPORT": 20, "WARRANTY": 21, "NaN": 22}
LAPTOP_ATTRIBUTES = {"CONNECTIVITY": 0, "DESIGN_FEATURES": 1, "GENERAL": 2, "MISCELLANEOUS": 3, "OPERATION_PERFORMANCE": 4,"PORTABILITY": 5, "PRICE": 6, "QUALITY": 7, "USABILITY": 8, "NaN": 9}
RESTAURANT_ENTITIES = {"AMBIENCE": 0, "DRINKS": 1, "FOOD": 2, "LOCATION": 3, "RESTAURANT": 4, "SERVICE": 5, "NaN": 6}
RESTAURANT_ATTRIBUTES = {"GENERAL": 0, "MISCELLANEOUS": 1, "PRICES": 2, "QUALITY": 3, "STYLE_OPTIONS": 4, "NaN": 5}


def load_datasets(use_restaurant):
    print(ROOT)
    #default load laptop data
    train_file = RAW_FILES[0]
    test_file = RAW_FILES[1]
    entities = LAPTOP_ENTITIES
    attributes = LAPTOP_ATTRIBUTES

    if use_restaurant:
        train_file = RAW_FILES[2]
        test_file = RAW_FILES[3]
        entities= RESTAURANT_ENTITIES
        attributes = RESTAURANT_ATTRIBUTES

    train_set = preprocess.load_data_as_df(os.path.join(RAW_DATA, train_file))
    test_set = preprocess.load_data_as_df(os.path.join(RAW_DATA, test_file))
    return train_set, test_set, entities, attributes

@plac.annotations(
    dataset=("Which dataset to use", "positional", None, str, ["restaurants", "laptop"]),
    label=("Which labels to use", "positional", None, str, ["entity", "attribute"]),
    embedding=("Which type of embeddings should be used", "option", None, str, ['glove', 'bert']),
    use_kcl=("Flag to indicat if KCL loss function should be used, otherwise MCL", "flag", None),
    binary=("Flag to if binary or multiclass model should be trained", "flag", "b"),
    binary_target_class=("Class to use as positive label in binary classifier setting", "option", "c", str, None),
    epochs=("The number of epoch to train with", "option", "e", int, None),
    lr=("Learning rate", "option", "lr", float, None),
    cuda=("Flag if cuda should be used", "flag", None)
)
def main(dataset="restaurants", label="entity", embedding='glove', use_kcl=False, binary=False, binary_target_class='GENERAL', epochs=1000, lr=0.0005, cuda=False):
    use_attributes = label == 'attribute' 
    use_restaurant = dataset == 'restaurants'

    embeddings = WordEmbeddings(embedding)
    embedding_dim = 100 if embedding == 'glove' else 3072

    #Loading data and correct labels
    train_set, test_set, entities, attributes = load_datasets(use_restaurant)
    # This is the dimension of the output of the ABAE model, the classification model gets this as input
    # It does not need to be related to the number of classes etc.
    output_dim = len(attributes if use_attributes else entities)
    
    if binary:
        train_dataset, other_train_dataset = dfToBinarySamplingDatasets(train_set, use_attributes, 
                                                                        binary_target_class, embeddings)
        test_dataset, other_test_dataset = dfToBinarySamplingDatasets(test_set, use_attributes, 
                                                                        binary_target_class, embeddings)
        print(len(train_dataset), len(other_train_dataset))
    else:
        train_dataset = dfToDataset(train_set, entities, attributes, embeddings)
        test_dataset = dfToDataset(test_set, entities, attributes, embeddings)
        print(len(train_dataset))

    print("Loaded dataset")
    


    param = {
        "dataset":dataset,
        "label": label,
        "embedding": embedding,
        "binary":binary,
        "binary_target_class": binary_target_class,
        "embedding_dim": embedding_dim,
        "output_dim": output_dim,
        "classification_dim": len(attributes if use_attributes else entities) if not binary else 1,
        "epochs": epochs,
        "lr": lr,
        "lr_decay_epochs": 350,
        "batch_size": 512,
        "use_padding": True,
        "validation_percentage": 0.1,
        "binary_sampling_percentage": 1, # should this be 1???
        "cuda": cuda,
        "use_kcl": use_kcl,
        "with_supervised": False,
        "use_micro_average": True,
        "train_entities": True,
        "save_training_records": True,
        "records_data_path": 'records/{}/{}/'.format(dataset, label)
    }

    if binary:
        trainer = Trainer(train_dataset, param, other_train_dataset)
    else:
        trainer = Trainer(train_dataset, param)
    model = trainer.train()
    model = trainer.train_classifier(freeze=False, new_param=param)

if __name__ == '__main__':
    print(plac.call(main))