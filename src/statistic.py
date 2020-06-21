import json
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

METADATA_KEYS = ['dataset',
 'label',
 'embedding',
 'binary',
 'binary_target_class',
 'embedding_dim',
 'output_dim',
 'classification_dim',
 'epochs',
 'lr',
 'lr_decay_epochs',
 'batch_size',
 'use_padding',
 'validation_percentage',
 'binary_sampling_percentage',
 'cuda',
 'use_kcl',
 'with_supervised',
 'use_micro_average',
 'train_entities',
 'target_class',
 'freeze',
 'save_training_records',
 'use_linmodel',
 'switch_to_relu',
 'records_data_path']
LOSS_KEYS = ['epoch', 'model', 'loss', 'eval_loss']
PERFORMANCE_KEYS = ['f1', 'recall', 'precision', 'epoch', 'step', 'model']


def load_log(path):
    with open(path) as f:
        data = {
            'filename' : os.path.basename(path),
            'abae_losses' : [],
            'classifier_losses':[],
            'classifier_train_performance' : [],
            'classifier_eval_performance' : [],
        }
        for line in f:
            #loading line of log
            log = json.loads(line)
            #getting the keys contained
            keys = list(log.keys())
            
            #organizing the logs based on keys and values
            if set(keys) >= set(METADATA_KEYS):
                #saving all metadata as key value pairs
                data.update(log)
            elif keys == LOSS_KEYS:
                if log['model'] == 'classifier':
                    data['classifier_losses'].append(log)
                else:
                    data['abae_losses'].append(log)
            elif keys == PERFORMANCE_KEYS:
                if log['step'] == 'train':
                    data['classifier_train_performance'].append(log)
                else:
                    data['classifier_eval_performance'].append(log)
    return data

def load_logs(path, files):
    all_logs = []
    for file in files:
        all_logs.append(load_log(path/file))
    return all_logs

def load_logs_as_df(path, files):
    return pd.DataFrame(load_logs(path, files))

def get_rows(df, selection_criteria):
    df_selection = df.copy()
    for key, value in selection_criteria.items():
        if value is None:
            continue
        elif isinstance(value, list):
            df_selection = df_selection[df_selection[key].isin(value)]
        else:
            df_selection = df_selection[df_selection[key] == value]
    return df_selection

def calculate_statistics(df, difference, metric='f1', pretraining=False, subplot=None, selection=None, filter=True):
    df.sort_values(difference, inplace=True)

    abae_df = None
    train_df = None
    eval_df = None
    for index, row in df.iterrows():
        abae = pd.DataFrame(row['abae_losses']).drop_duplicates()
        abae[difference] = row[difference]
        min_abae = abae[abae.loss == abae.loss.min()]
        if abae_df is None:
            abae_df = min_abae
        else:
            abae_df = abae_df.append(min_abae)

        train = pd.DataFrame(row['classifier_train_performance']).drop_duplicates()
        train[difference] = row[difference]
        max_train = train[train.f1 == train.f1.max()]
        max_train = max_train[max_train.epoch == max_train.epoch.min()]
        if train.f1.max() > 0.0:
            if train_df is None:
                train_df = max_train
            else:
                train_df = train_df.append(max_train)

        eval = pd.DataFrame(row['classifier_eval_performance']).drop_duplicates()
        eval[difference] = row[difference]
        max_eval = eval[eval.f1 == eval.f1.max()]
        max_eval = max_eval[max_eval.epoch == max_eval.epoch.min()]
        if eval.f1.max() > 0.0:
            if eval_df is None:
                eval_df = max_eval
            else:
                eval_df = eval_df.append(max_eval)

    # Now we want to calculate some statistics for all combinations in difference
    if subplot is None:
        fig, subplot = plt.subplots()

    if pretraining:
        abae_df = abae_df.groupby(difference)
        loss = abae_df.eval_loss.tolist()
        subplot.boxplot(loss)
    else:
        data = []
        labels = []
        color = []
        try:
            difference_values = train_df[difference].unique()
            for value in difference_values:
                train = train_df[train_df[difference] == value].f1.tolist()
                data.append(train)
                labels.append("t: "+ str(value))
                color.append(['lightskyblue', 'blue'])
        except TypeError:
            if difference == "use_kcl":
                r = 2
            for _ in range(r):
                data.append([])
                labels.append("")
                color.append(['lightskyblue', 'blue'])
        try:
            difference_values = eval_df[difference].unique()
            for value in difference_values:
                eval = eval_df[eval_df[difference] == value].f1.tolist()
                data.append(eval)
                labels.append("v: " + str(value))
                color.append(['tomato', 'red'])
        except TypeError:
            if difference == "use_kcl":
                r = 2
            for _ in range(r):
                data.append([])
                labels.append("")
                color.append(['lightskyblue', 'blue'])
        bp = subplot.boxplot(data, labels=labels, patch_artist=True)
        for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians']:
            for i, elem in enumerate(bp[element]):
                if element == 'whiskers':
                    elem.set(color=color[int(i/2)][1])
                else:
                    elem.set(color=color[i][1])
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor(color[i][0])

        if selection is not None:
            subplot.set_title(selection['binary_target_class'] + ": " + difference)
            subplot.set_ylabel('F1')


def calculate_lr_statistics(df, difference, metric='f1', pretraining=False, subplot=None, selection=None, filter=True):
    df.sort_values(difference, inplace=True)
    
    abae_df = None
    train_df = None
    eval_df = None
    for index, row in df.iterrows():
        abae = pd.DataFrame(row['abae_losses']).drop_duplicates()
        abae[difference] = row[difference]
        min_abae = abae[abae.loss == abae.loss.min()]
        if abae_df is None:
            abae_df = min_abae
        else:
            abae_df = abae_df.append(min_abae)

        train = pd.DataFrame(row['classifier_train_performance']).drop_duplicates()
        train[difference] = row[difference]
        max_train = train[train.f1 == train.f1.max()]
        max_train = max_train[max_train.epoch == max_train.epoch.min()]
        #if train.f1.max() > 0.0:
        if train_df is None:
            train_df = max_train
        else:
            train_df = train_df.append(max_train)

        eval = pd.DataFrame(row['classifier_eval_performance']).drop_duplicates()
        eval[difference] = row[difference]
        max_eval = eval[eval.f1 == eval.f1.max()]
        max_eval = max_eval[max_eval.epoch == max_eval.epoch.min()]
        #if eval.f1.max() > 0.0:
        if eval_df is None:
            eval_df = max_eval
        else:
            eval_df = eval_df.append(max_eval)

    # Now we want to calculate some statistics for all combinations in difference
    if subplot is None:
        fig, subplot = plt.subplots()

    data = []
    labels = []
    color = []
    difference_values = train_df[difference].unique()
    try:
        for value in difference_values:
            train = train_df[train_df[difference] == value].f1.tolist()
            data.append(train)
            labels.append(str(value))
            color.append(['lightskyblue', 'blue'])
        for i in range(len(difference_values), 4):
            data.append([])
            labels.append("")
            color.append(['lightskyblue', 'blue'])
    except TypeError:
        if difference == "lr":
            r = 4
        for _ in range(r):
            data.append([])
            labels.append("")
            color.append(['lightskyblue', 'blue'])
    try:
        difference_values = eval_df[difference].unique()
        for value in difference_values:
            eval = eval_df[eval_df[difference] == value].f1.tolist()
            data.append(eval)
            labels.append(str(value))
            color.append(['tomato', 'red'])
        for i in range(len(difference_values), 4):
            data.append([])
            labels.append("")
            color.append(['tomato', 'red'])
    except TypeError:
        if difference == "lr" and not selection['train_entities']:
            r = 4
        else:
            r = 2
        for _ in range(r):
            data.append([])
            labels.append("")
            color.append(['tomato', 'red'])
    
    bp = subplot.boxplot(data, labels=labels, patch_artist=True)
    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians']:
        for i, elem in enumerate(bp[element]):
            if element == 'whiskers':
                elem.set(color=color[int(i/2)][1])
            else:
                elem.set(color=color[i][1])
    for i, box in enumerate(bp['boxes']):
        box.set_facecolor(color[i][0])
    
    if selection is not None:
            subplot.set_title(selection['binary_target_class'] + ": " + difference)
            subplot.set_ylabel('F1')


if __name__ == "__main__":
    ROOT = Path(os.getcwd())
    DATA = ROOT/'data'
    SRC =  ROOT/'src'
    RAW_DATA = DATA/'raw'
    RECORDS = ROOT/'records'
    RESTAURANT_ATTRIBUTES = RECORDS/'restaurants'/'attribute'
    RESTAURANT_ENTITIES = RECORDS/'restaurants'/'entity'
    ATTR_FILES = os.listdir(RESTAURANT_ATTRIBUTES)
    ENT_FILES = os.listdir(RESTAURANT_ENTITIES)
    if '.gitignore' in ATTR_FILES:
        ATTR_FILES.remove('.gitignore')
    if '.gitignore' in ENT_FILES:
        ENT_FILES.remove('.gitignore')
    attr_logs = load_logs_as_df(RESTAURANT_ATTRIBUTES, ATTR_FILES)
    ent_logs = load_logs_as_df(RESTAURANT_ENTITIES, ENT_FILES)
    logs = pd.concat([attr_logs, ent_logs])
    SELECTION = {
        'dataset':'restaurants', 
        'train_entities':False, 
        'embedding':'bert-base-cased', 
        #'use_kcl': [True, False], 
        #'lr':[0.0005, 0.00005], 
        'binary':True, 
        'binary_target_class' : 'GENERAL'
    }
    rows = get_rows(logs, SELECTION)
    calculate_lr_statistics(rows, 'lr', metric='f1', selection=SELECTION)