import os
import glob
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import flair
from flair.embeddings import BertEmbeddings

flair.device = torch.device('cpu')

from sklearn.metrics import f1_score, precision_score, recall_score

from Model import LinModel, Classification, load_model
import preprocess
from Dataset import AspectDataset, dfToBinarySamplingDatasets, dfToDataset, collate_padding

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

def calculate_metrics(targets, predictions, average='binary'):
    """ Calculates common performance metrics. 
    Arguments:
        targets {torch.tensor[N]} -- The target values for a given sample.
        predictions {torch.tensor[N x output_dim]} -- The softmax/sigmoid output for
            that sample
        average {String} -- Which score average to use. Possible: binary, micro, macro, samples
    Returns:
        {dict{f1, recall, precision}}
    """
    # If the output dimension is 1, we used the sigmoid function
    # We want to compute for each prediction the argmax class -> for sigmoid
    statistic = {}
    max_classes = np.squeeze(predictions)
    statistic["f1"] = f1_score(targets, max_classes, average=average)
    statistic["recall"] = recall_score(targets, max_classes, average=average)
    statistic["precision"] = precision_score(targets, max_classes, average=average)
    return statistic

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

""" Parameters to set: """
use_attributes = False
log_folder_string = "records/restaurants/entity_multiple_runs"
""" End parameters """

model_folder_string = "models/restaurants/" + ("attribute/" if use_attributes else "entity/")
log_files = glob.glob(log_folder_string+"/training*")
print("Evaluation of:", "attribute" if use_attributes else "entity")

device = torch.device('cpu')

aggregated_targets = []
aggregated_outputs = []

_, test_set, entities, attributes = load_datasets(True)

for file in log_files:

    model_name = file.split("\\")[-1]
    with open(file, "r") as f:
        line = f.readline()
        param = json.loads(line)

    if "binary_target_class" not in param.keys():
        print("Encountered multiclass training...")
        print("Skipping:", file)
        continue
    if "freeze" in param.keys() and param["freeze"]:
        print("Encountered freezed model...")
        print("Skipping:", file)
        continue
    binary_target_class = param["binary_target_class"]

    embeddings = BertEmbeddings("bert-base-cased")
    embedding_dim = 3072
    print("Loaded embeddings")

    middel_dim = 6 if use_attributes else 7

    model = LinModel(embedding_dim, middel_dim)
    if "activation" in param.keys():
        activation = param["activation"]
    else:
        activation = "softmax"
    model = Classification(model, middel_dim, output_dim=1, activation=activation)
    try:
        model = load_model(model, model_folder_string+model_name)
    except:
        print(model_folder_string)
        print("Could not find model file:", model_folder_string+model_name)
        continue
    #model.to(device)
    print("Loaded model")

    train_dataset, other_train_dataset = dfToBinarySamplingDatasets(test_set, use_attributes, 
                                                                            binary_target_class, embeddings)

    print("Loaded dataset for:", binary_target_class)

    dataloader = DataLoader(train_dataset, batch_size=2048, collate_fn=collate_padding)
    other_dataloader = DataLoader(other_train_dataset, batch_size=2048, collate_fn=collate_padding)

    print("Getting Predictions...")
    for sentences, entities, attributes in dataloader:
        if use_attributes:
            target = attributes
        else:
            target = entities
        print(sentences.device)
        #sentences = sentences.to(device)
        output = model(sentences)
        output.detach()
        if output.size()[1] == 1:
            max_classes = torch.round(output)
        else:
            max_classes = torch.argmax(output, dim=1)
        max_classes = max_classes.to(torch.device('cpu')).detach().numpy()
        targets = target.to(torch.device('cpu')).detach().numpy()

        aggregated_targets.append(targets)
        aggregated_outputs.append(max_classes)
        del sentences

    for sentences, entities, attributes in other_dataloader:
        if use_attributes:
            target = attributes
        else:
            target = entities
        #sentences = sentences.to(device)
        output = model(sentences)
        output.detach()
        if output.size()[1] == 1:
            max_classes = torch.round(output)
        else:
            max_classes = torch.argmax(output, dim=1)
        max_classes = max_classes.to(torch.device('cpu')).detach().numpy()
        targets = target.to(torch.device('cpu')).detach().numpy()

        aggregated_targets.append(targets)
        aggregated_outputs.append(max_classes)
        del sentences

    model = model.to(torch.device("cpu"))
    del model

aggregated_targets = np.concatenate(aggregated_targets)
aggregated_outputs = np.concatenate(aggregated_outputs)
metrics = calculate_metrics(aggregated_targets, aggregated_outputs)
print(metrics)