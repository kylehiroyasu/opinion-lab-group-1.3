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

""" Parameters to set: """
attribute_folder_string = "records/restaurants/attribute_multiple_runs_kcl"
entity_folder_string = "records/restaurants/entity_multiple_runs_kcl"
""" End parameters """

log_files = glob.glob(attribute_folder_string+"/training*")
log_files += glob.glob(entity_folder_string+"/training*")
print("Evaluation of: both")

device = torch.device('cpu')

_, test_set, entities, attributes = load_datasets(True)
print("Loaded dataset")

embeddings = BertEmbeddings("bert-base-cased")
embedding_dim = 3072
print("Loaded embeddings")

dataset = dfToDataset(test_set, entities, attributes, embeddings)
dataloader = DataLoader(dataset, 1000, shuffle=False, collate_fn=collate_padding)
# in predictions we will save each sentence with: [true_class: (Attribute, Entity), attribute_pred: [], entity_pred: []]
predictions = []
for _, ent, att in dataloader:
    for index in range(len(ent)):
        ent_target = ent[index].item()
        att_target = att[index].item()
        ent_pred_list = [0.0 for ent in entities]
        att_pred_list = [0.0 for att in attributes]
        predictions.append([(att_target, ent_target), att_pred_list, ent_pred_list])

print("Start Evaluation")
for file in log_files:

    model_name = os.path.basename(file)
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
    use_attributes = param["label"] == "attribute"

    middel_dim = 6 if use_attributes else 7

    model_folder_string = "models/restaurants/" + ("attribute/" if use_attributes else "entity/")
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
    
    start_index = 0
    for sentences, ent, att in dataloader:
        output = model(sentences)
        output.detach()
        for index in range(len(ent)):
            prob = output[index].item()
            if use_attributes:
                # [Sentence][Attributes][Target]
                predictions[start_index+index][1][attributes[binary_target_class]] += prob
            else:
                # [Sentence][Entities][Target]
                predictions[start_index+index][2][entities[binary_target_class]] += prob
        start_index += len(entities)

    model = model.to(torch.device("cpu"))
    del model


revised_predictions = [[], []]
for t, att, ent in predictions:
    a_t, e_t = t
    target = str(a_t)+"#"+str(e_t)
    revised_predictions[0].append(target)
    max_att = att.index(max(att))
    max_ent = ent.index(max(ent))
    max_prediction = str(max_att)+"#"+str(max_ent)
    revised_predictions[1].append(max_prediction)

target = np.array(revised_predictions[0])
predictions = np.array(revised_predictions[1])

statistic = {}
average = "micro"
statistic["f1"] = f1_score(target, predictions, average=average)
statistic["recall"] = recall_score(target, predictions, average=average)
statistic["precision"] = precision_score(target, predictions, average=average)

print(statistic)