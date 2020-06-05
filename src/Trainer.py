import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from Model import Model, LinModel, Classification, save_model
from Dataset import collate_padding
from Loss import KCL, MCL, Class2Simi
from Learners import Learner_Clustering

verbose_training = True


class Trainer:

    def __init__(self, dataset, param_dict, other_dataset):
        """ Initializes a Trainer object which will be used to optimize
        the neural network.
        Arguments:
            dataset {AspectDataset} -- the part of the data usable for training and validation
            param_dict {dict} -- parameter dictionary, parameters can be seen above
            other_dataset {AspectDataset} -- dataset create by dfToBinarySamplingDatasets. Used
            if we want to vary the number of similar (in dataset) and dissimilar samples (in 
            other_dataset). We can only use this in the binary classification setting.
        """
        self.dataset = dataset
        self.other_dataset = other_dataset
        self.param = param_dict
        if param_dict["use_linmodel"]:
            self.model = LinModel(self.param["embedding_dim"], self.param["output_dim"])
        else:
            self.model = Model(self.param["embedding_dim"], self.param["output_dim"])
        # Next value is used for iterting through the other_dataset in binary_sampling,
        # see getOtherBatch()
        self.use_train_iterator = True
        self.other_iterator = None
        # Use for the classification training afterwards
        self.only_supervised = False

        self.filename = 'training_{}'.format(
            datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        self.training_records = [self.param]
        self.model_name = 'binary-abae'
        if self.param["save_model_path"] is not None:
            self.model_path = os.path.join(os.getcwd(), self.param["save_model_path"], self.filename)
        self.current_epoch = 0
        self.best_train_f1 = 0.0
        self.best_eval_f1 = 0.0

    def train(self, verbose=True):
        """ Starts the training procedure.
        Arguments:
            verbose {bool} -- Whether to log messages during training
        Returns:
            {Model} -- The trained model
        """

        verbose_training = verbose

        train_dataset, validation_dataset = split_dataset(
            self.dataset, self.param["validation_percentage"])
        other_train_dataset, other_val_dataset = split_dataset(
            self.other_dataset, self.param["validation_percentage"])

        # Create the dataloaders for sampling, as we use the binary case we additionally intialize dataloaders for the
        # other classes
        self.dataloader = DataLoader(
            train_dataset, batch_size=self.param["batch_size"], shuffle=True, collate_fn=collate_padding)
        self.validloader = DataLoader(
            validation_dataset, batch_size=self.param["batch_size"], collate_fn=collate_padding)
        batch_size = int(round(self.param["batch_size"]*self.param["binary_sampling_percentage"]))
        self.other_dataloader = DataLoader(
            other_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_padding)
        self.other_validloader = DataLoader(
            other_val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_padding)

        # Initialize the optimizer, Learning rate scheduler and the classification loss
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param["lr"])
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.param["lr"])
        self.scheduler = StepLR(self.optimizer, step_size=120, gamma=0.1)
        self.classification_loss = nn.BCELoss()
        
        # Initiliaze the correct loss. This is wrapped by the learner object which takes care
        # of all the similarity calculations
        if self.param["use_kcl"]:
            self.learner_clustering = Learner_Clustering(KCL())
        else:
            self.learner_clustering = Learner_Clustering(MCL())

        if self.param["cuda"] and torch.cuda.is_available():
            log("Using GPU")
            device = torch.device('cuda')
            self.model.to(device)
        else:
            log("Using CPU")
            self.param["cuda"] = False
            device = torch.device('cpu')

        # The patience value will be used to determine whethter we want to stop the training early
        # In each epoch with the validation error not decreasing patience will be decreased.
        # If it as a zero, the training will be terminated.
        patience = self.param["patience_early_stopping"]
        best_eval_loss = torch.tensor(float('inf'))

        for e in range(self.param["epochs"]):
            self.current_epoch = e
            log("Epoch:", e)

            # Start one training epoch and log the loss
            self.model.train()
            loss = self.train_epoch()
            loss = loss.to(torch.device('cpu'))
            log("Train loss:", loss.item())

            # Start one evaluation epoch and log the loss
            self.model.eval()
            eval_loss = self.eval_epoch()
            eval_loss = eval_loss.to(torch.device('cpu'))
            log("Eval Loss:", eval_loss.item())
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                patience = self.param["patience_early_stopping"]
                if self.param["save_model_path"] is not None:
                    save_model(self.model, self.model_path)
            else:
                patience -= 1
            log("Current patience:", patience)
            if patience <= 0:
                break

            # We might not want to decay the learning rate during the whole training
            if e < self.param["lr_decay_epochs"]:
                self.scheduler.step()

            self.training_records.append(
                {'epoch': e, 'model': self.model_name, 'loss': loss.item(), 'eval_loss': eval_loss.item()})
        if self.param["save_training_records"]:
            save_records(self.param['records_data_path'],
                         self.filename, self.training_records)
        if self.param["save_model_path"] is not None:
            print("Reloading best model")
            self.model.load_state_dict(torch.load(self.model_path))
        if self.only_supervised:
            log("Best Scores:", self.best_train_f1, "Train F1", self.best_eval_f1, "Validation F1")
        return self.model

    def train_classifier(self, freeze=True, new_param=None):
        """ Trains a classification layer on top of the previously trained model.
        The parameters of the previous migh the freezed
        Arguments:
            freeze {bool} -- whether to freeze the previous parameters
        Returns:
            {Model} -- the final model
        """
        self.model_name = 'classifier'
        if new_param is not None:
            self.param = new_param
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.only_supervised = True
        self.model = Classification(self.model, self.param["output_dim"], self.param["classification_dim"])
        model = self.train()
        return model

    def train_epoch(self):
        """ Training method, if we want to only predict one class -> transformation of
        the problem to binary classification.
        The parameters for training are all stored in self.param
        """
        aggregated_targets = []
        aggregated_outputs = []
        loss = torch.zeros(1)
        for sentences, entities, attributes in self.dataloader:
            # After getting a batch from the other classes, simply append them to the current
            # sentences. The error calculation is robust enough.
            other_sentences, other_entities, other_attributes = self.getOtherBatch(train=True)
            sentences = self.fuze_sentences(sentences, other_sentences)
            entities = torch.cat([entities, other_entities])
            attributes = torch.cat([attributes, other_attributes])

            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()

            batch_loss, output = self.train_batch(sentences, target)
            loss += batch_loss
            aggregated_targets.append(target.to(torch.device('cpu')))
            aggregated_outputs.append(output.to(torch.device('cpu')))
        aggregated_targets = torch.cat(aggregated_targets)
        aggregated_outputs = torch.cat(aggregated_outputs)
        # Only if we are in the classification phase we can get metrics
        if self.only_supervised:
            metrics = calculate_metrics(aggregated_targets, aggregated_outputs)
            if metrics["f1"] > self.best_train_f1:
                self.best_train_f1 = metrics["f1"]
            log("Train", metrics)
            metrics.update({
                'epoch': self.current_epoch,
                'step': 'train',
                'model': self.model_name})
            self.training_records.append(metrics)

        return loss

    def train_batch(self, sentences, target):
        """ Training method for one data batch.
        """
        self.optimizer.zero_grad()

        output = self.model(sentences)

        # only_supervised means we are training the classifier after the ABAE model
        if not self.only_supervised:
            similarity = Class2Simi(target)
            loss = self.learner_clustering.calculate_criterion(
                output, similarity)
        else:
            # In the binary case we need to add one dimension
            loss = self.classification_loss(output, target[:,None])

        loss.backward()
        self.optimizer.step()
        return loss, output

    def eval_epoch(self):
        aggregated_targets = []
        aggregated_outputs = []
        loss = torch.tensor(0.0)
        for sentences, entities, attributes in self.validloader:
            # After getting a batch from the other classes, simply append them to the current
            # sentences. The error calculation is robust enough.
            other_sentences, other_entities, other_attributes = self.getOtherBatch(train=False)
            # Combining the sample depends on wheter we used padding or not (=tensor vs list output of dataloader)
            sentences = self.fuze_sentences(sentences, other_sentences)
            entities = torch.cat([entities, other_entities])
            attributes = torch.cat([attributes, other_attributes])

            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()

            batch_loss, output = self.eval_batch(sentences, target)
            loss += batch_loss
            aggregated_targets.append(target.to(torch.device('cpu')))
            aggregated_outputs.append(output.to(torch.device('cpu')))
        aggregated_targets = torch.cat(aggregated_targets)
        aggregated_outputs = torch.cat(aggregated_outputs)
        # Only if we are in the classification case we can get metrics
        if self.only_supervised:
            metrics = calculate_metrics(aggregated_targets, aggregated_outputs)
            if metrics["f1"] > self.best_eval_f1:
                self.best_eval_f1 = metrics["f1"]
            log("Eval", metrics)
            metrics.update({
                'epoch': self.current_epoch,
                'step': 'eval',
                'model': self.model_name})
            self.training_records.append(metrics)
        return loss

    def eval_batch(self, sentences, target):
        """ Evaluation of one batch. """
        output = self.model(sentences)

        if not self.only_supervised:
            similarity = Class2Simi(target)
            loss = self.learner_clustering.calculate_criterion(
                output, similarity)
        else:
            # In the binary case we need to add one dimension
            loss = self.classification_loss(output, target[:,None])
        return loss, output

    def fuze_sentences(self, sentences, other_sentences):
        """ Combines the sentences and other_sentences into one tensor, which has the maximum sentence
        length as second dimension 
        Arguments:
            sentences {tensor[batch1, max_sentence_length, embedding_dim]}
            other_sentences {tensor[batch2, max_other_sentence_length, embedding_dim]}
        Returns:
            {tensor[batch1+batch2, max(max_sentence_length, max_other_sentence_length), embedding_dim]}
        """
        sentences_max_length = sentences.size()[1]
        other_sentences_max_length = other_sentences.size()[1]
        # We need to check which tensor needs additional padding before we can concatenate them
        if sentences_max_length > other_sentences_max_length:
            new_size = other_sentences.size()[0], sentences_max_length, other_sentences.size()[2]
            new_other = torch.zeros(new_size, device=other_sentences.device)
            new_other[:, :other_sentences_max_length,:] = other_sentences
            other_sentences = new_other
        elif sentences_max_length < other_sentences_max_length:
            new_size = sentences.size()[0], other_sentences_max_length, sentences.size()[2]
            new_sentences = torch.zeros(new_size, device=sentences.device)
            new_sentences[:, :sentences_max_length, :] = sentences
            sentences = new_sentences
        return torch.cat([sentences, other_sentences])

    def getOtherBatch(self, train):
        """ This method basically gives you the next batch of samples from the other classes
        in the binary classificationi case. It is saving the state of a iterator and if 
        that iterator is done, gets a new one. The datasets must be non-empty. """
        # We already have an iterator and the iterator is for the same dataset.
        if self.use_train_iterator == train and self.other_iterator is not None:
            try:
                return self.other_iterator.__next__()
            except StopIteration:
                # If there are no elements left, we just create a new iterator
                pass
        if train:
            dataloader = self.other_dataloader
        else:
            dataloader = self.other_validloader
        self.use_train_iterator = train
        self.other_iterator = dataloader.__iter__()
        return self.other_iterator.__next__()


class MulticlassTrainer(Trainer):

    def __init__(self, dataset, param_dict):
        super(MulticlassTrainer, self).__init__(dataset, param_dict, None)

    def train(self, verbose=True):
        """ Starts the training procedure.
        Arguments:
            verbose {bool} -- Whether to log messages during training
        Returns:
            {Model} -- The trained model
        """
        verbose_training = verbose

        train_dataset, validation_dataset = split_dataset(
            self.dataset, self.param["validation_percentage"])

        # Create the dataloaders for sampling, as we use the binary case we additionally intialize dataloaders for the
        # other classes
        self.dataloader = DataLoader(
            train_dataset, batch_size=self.param["batch_size"], shuffle=True, collate_fn=collate_padding)
        self.validloader = DataLoader(
            validation_dataset, batch_size=self.param["batch_size"], collate_fn=collate_padding)

        # Initialize the optimizer, Learning rate scheduler and the classification loss
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param["lr"])
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.param["lr"])
        self.scheduler = StepLR(self.optimizer, step_size=120, gamma=0.1)
        self.classification_loss = nn.CrossEntropyLoss()
        
        # Initiliaze the correct loss. This is wrapped by the learner object which takes care
        # of all the similarity calculations
        if self.param["use_kcl"]:
            self.learner_clustering = Learner_Clustering(KCL())
        else:
            self.learner_clustering = Learner_Clustering(MCL())

        if self.param["cuda"] and torch.cuda.is_available():
            log("Using GPU")
            device = torch.device('cuda')
            self.model.to(device)
        else:
            log("Using CPU")
            self.param["cuda"] = False
            device = torch.device('cpu')

        # The patience value will be used to determine whethter we want to stop the training early
        # In each epoch with the validation error not decreasing patience will be decreased.
        # If it as a zero, the training will be terminated.
        patience = self.param["patience_early_stopping"]
        best_eval_loss = torch.tensor(float('inf'))

        for e in range(self.param["epochs"]):
            self.current_epoch = e
            log("Epoch:", e)

            # Start one training epoch and log the loss
            self.model.train()
            loss = self.train_epoch()
            loss = loss.to(torch.device('cpu'))
            log("Train loss:", loss.item())

            # Start one evaluation epoch and log the loss
            self.model.eval()
            eval_loss = self.eval_epoch()
            eval_loss = eval_loss.to(torch.device('cpu'))
            log("Eval Loss:", eval_loss.item())
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                patience = self.param["patience_early_stopping"]
                if self.param["save_model_path"] is not None:
                    save_model(self.model, self.model_path)
            else:
                patience -= 1
            log("Current patience:", patience)
            if patience <= 0:
                break

            # We might not want to decay the learning rate during the whole training
            if e < self.param["lr_decay_epochs"]:
                self.scheduler.step()

            self.training_records.append(
                {'epoch': e, 'model': self.model_name, 'loss': loss.item(), 'eval_loss': eval_loss.item()})
        if self.param["save_training_records"]:
            save_records(self.param['records_data_path'],
                         self.filename, self.training_records)
        if self.param["save_model_path"] is not None:
            print("Reloading best model")
            self.model.load_state_dict(torch.load(self.model_path))
        if self.only_supervised:
            log("Best Scores:", self.best_train_f1, "Train F1", self.best_eval_f1, "Validation F1")
        return self.model

    def train_epoch(self):
        """ Training method, if we want to only predict one class -> transformation of
        the problem to binary classification.
        The parameters for training are all stored in self.param
        """
        aggregated_targets = []
        aggregated_outputs = []
        loss = torch.zeros(1)
        for sentences, entities, attributes in self.dataloader:
            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()

            batch_loss, output = self.train_batch(sentences, target)
            loss += batch_loss
            aggregated_targets.append(target.to(torch.device('cpu')))
            aggregated_outputs.append(output.to(torch.device('cpu')))
        aggregated_targets = torch.cat(aggregated_targets)
        aggregated_outputs = torch.cat(aggregated_outputs)
        # Only if we are in the classification phase we can get metrics
        if self.only_supervised:
            metrics = calculate_metrics(aggregated_targets, aggregated_outputs, average='micro')
            if metrics["f1"] > self.best_train_f1:
                self.best_train_f1 = metrics["f1"]
            log("Train", metrics)
            metrics.update({
                'epoch': self.current_epoch,
                'step': 'train',
                'model': self.model_name})
            self.training_records.append(metrics)

        return loss

    def train_batch(self, sentences, target):
        """ Training method for one data batch.
        """
        self.optimizer.zero_grad()

        output = self.model(sentences)

        # only_supervised means we are training the classifier after the ABAE model
        if not self.only_supervised:
            similarity = Class2Simi(target)
            loss = self.learner_clustering.calculate_criterion(
                output, similarity)
        else:
            loss = self.classification_loss(output, target)

        loss.backward()
        self.optimizer.step()
        return loss, output

    def eval_epoch(self):
        aggregated_targets = []
        aggregated_outputs = []
        loss = torch.tensor(0.0)
        for sentences, entities, attributes in self.validloader:
            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()

            batch_loss, output = self.eval_batch(sentences, target)
            loss += batch_loss
            aggregated_targets.append(target.to(torch.device('cpu')))
            aggregated_outputs.append(output.to(torch.device('cpu')))
        aggregated_targets = torch.cat(aggregated_targets)
        aggregated_outputs = torch.cat(aggregated_outputs)
        # Only if we are in the classification case we can get metrics
        if self.only_supervised:
            metrics = calculate_metrics(aggregated_targets, aggregated_outputs, average='micro')
            if metrics["f1"] > self.best_eval_f1:
                self.best_eval_f1 = metrics["f1"]
            log("Eval", metrics)
            metrics.update({
                'epoch': self.current_epoch,
                'step': 'eval',
                'model': self.model_name})
            self.training_records.append(metrics)
        return loss

    def eval_batch(self, sentences, target):
        """ Evaluation of one batch. """
        output = self.model(sentences)

        if not self.only_supervised:
            similarity = Class2Simi(target)
            loss = self.learner_clustering.calculate_criterion(
                output, similarity)
        else:
            loss = self.classification_loss(output, target)
        return loss, output

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
    if predictions.size()[1] == 1:
        max_classes = torch.round(predictions)
    else:
        max_classes = torch.argmax(predictions, dim=1)
    max_classes = max_classes.to(torch.device('cpu')).detach().numpy()
    targets = targets.to(torch.device('cpu')).detach().numpy()
    statistic = {}
    max_classes = np.squeeze(max_classes)
    statistic["f1"] = f1_score(targets, max_classes, average=average)
    statistic["recall"] = recall_score(targets, max_classes, average=average)
    statistic["precision"] = precision_score(targets, max_classes, average=average)
    return statistic


def split_dataset(dataset, validation_percentage):
    """ Returns two datasets. One for training and the other one for validation.
    Arguments:
        dataset {torch.Dataset} -- the dataset to split
        validation_percentage {float} -- How much of the dataset shall be used for
            validation
    Returns:
        {(train: Dataset, validation: Dataset)}
    """
    assert (validation_percentage >= 0 and validation_percentage <= 1)
    validation_length = math.ceil(len(dataset) * validation_percentage)
    train_length = len(dataset) - validation_length
    datasetsList = random_split(dataset, [train_length, validation_length])
    return datasetsList[0], datasetsList[1]


def log(*string):
    if verbose_training:
        print(string)


def save_records(path, filename, records):
    with open(Path(os.getcwd())/path/filename, mode='a') as f:
        for line in records:
            f.write(json.dumps(line)+'\n')
