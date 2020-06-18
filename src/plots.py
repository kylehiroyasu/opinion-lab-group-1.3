import json
import os
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

SELECTION = {
    'dataset':'restaurants', 
    'train_entities':False, 
    'embedding':['glove','bert-base-cased'], 
    'use_kcl':True, 
    'lr':[0.0005, 0.00005], 
    'binary':True, 
    'binary_target_class' : 'GENERAL'
}


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


def plot_losses(df, difference):

    for index, row in df.iterrows():
        tmp = pd.DataFrame(row['classifier_losses'])

        differentiator = row[difference]
        epochs = tmp.epoch.tolist()
        train_loss = tmp.loss.tolist()
        eval_loss = tmp.eval_loss.tolist()

        plt.plot(epochs, train_loss, label='{} {}'.format(differentiator, 'train_loss'))
        plt.plot(epochs, eval_loss, label='{} {}'.format(differentiator, 'eval_loss'))

    plt.legend()
    plt.show()


def plot_performance(df, difference, metric='f1', subplot=None, title=None):

    if subplot is None:
        fig, subplot = plt.subplots()

    df.sort_values(difference, inplace=True)

    for index, row in df.iterrows():
        train_df = pd.DataFrame(row['classifier_train_performance'])
        eval_df = pd.DataFrame(row['classifier_eval_performance'])

        differentiator = row[difference]
        train_epochs = train_df.epoch.tolist()
        train_perf = train_df[metric].tolist()
        eval_epochs = eval_df.epoch.tolist()
        eval_perf = eval_df[metric].tolist()

        subplot.plot(train_epochs, train_perf, label='{} train_{}'.format(differentiator, metric))
        subplot.plot(eval_epochs, eval_perf, label='{} eval_{}'.format(differentiator, metric))

    subplot.legend()
    if title is None:
        subplot.set_title('{}-Scores Comparing {}'.format(metric, difference))
    else:
        subplot.set_title(title)