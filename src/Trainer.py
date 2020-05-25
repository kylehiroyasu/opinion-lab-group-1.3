import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score 

from Model import Model, Classification
from Dataset import collate, collate_padding
from Loss import KCL, MCL, Class2Simi
from Learners import Learner_Classification, Learner_Clustering

param = {
    "embedding_dim": 100,
    "output_dim": 7,
    "epochs": 40,
    "lr": 0.0025,
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

verbose_training = True

class Trainer:

    def __init__(self, dataset, param_dict, other_dataset=None):
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
        self.binary_sampling = self.other_dataset is not None
        self.param = param_dict
        self.model = Model(self.param["embedding_dim"], self.param["output_dim"])
        # Next value is used for iterting through the other_dataset in binary_sampling,
        # see getOtherBatch()
        self.use_train_iterator = True
        self.other_iterator = None
        # Use for the classification training afterwards
        self.only_supervised = False
        entity_length, attribute_length = self.dataset.targetLengths()
        self.min_target = 0
        self.max_target = entity_length if self.param["train_entities"] else attribute_length

    def train(self, verbose=True):
        """ Starts the training procedure.
        Arguments:
            verbose {bool} -- Wheter to log log messages during training
        Returns:
            {Model} -- The trained model
        """

        verbose_training = verbose

        train_dataset, validation_dataset = split_dataset(self.dataset, self.param["validation_percentage"])
        if self.binary_sampling:
            other_train_dataset, other_val_dataset = split_dataset(self.other_dataset, self.param["validation_percentage"])

        # Create the dataloaders for sampling, if we have the binary case we additionally intialize dataloaders for the 
        # other classes
        if self.param["use_padding"]:
            collate_fn = collate_padding
        else:
            collate_fn = collate

        self.dataloader = DataLoader(train_dataset, batch_size=self.param["batch_size"], shuffle=True, collate_fn=collate_fn)
        self.validloader = DataLoader(validation_dataset, batch_size=self.param["batch_size"], collate_fn=collate_fn)
        if self.binary_sampling:
            batch_size = int(round(self.param["batch_size"]*self.param["binary_sampling_percentage"]))
            self.other_dataloader = DataLoader(other_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            self.other_validloader = DataLoader(other_val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.param["lr"])
        self.scheduler = StepLR(self.optimizer, step_size=100, gamma=0.25)
        if self.binary_sampling:
            self.learner_classification = Learner_Classification(nn.BCELoss())
        else:
            self.learner_classification = Learner_Classification(nn.CrossEntropyLoss())
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

        for e in range(self.param["epochs"]):
            log("Epoch:", e)
            self.model.train()
            if self.binary_sampling:
                loss = self.train_bs_epoch()
            else:
                loss = self.train_epoch()
            loss = loss.to(torch.device('cpu'))
            log("Train loss:", loss.item())
            
            self.model.eval()
            if self.binary_sampling:
                eval_loss = self.eval_bs_epoch()
            else:
                eval_loss = self.eval_epoch()
            eval_loss = eval_loss.to(torch.device('cpu'))
            log("Eval Loss:", eval_loss.item())
            # TODO How do we do the evaluation, if we are not in the supervised case? Assign output to majority label? Compute centroids?
            if e < self.param["lr_decay_epochs"]:
                self.scheduler.step()
        return self.model

    def train_classifier(self, freeze=True, new_param=None):
        """ Trains a classification layer on top of the previously trained model.
        The parameters of the previous migh the freezed
        Arguments:
            freeze {bool} -- whether to freeze the previous parameters
        Returns:
            {Model} -- the final model
        """
        if new_param is not None:
            self.param = new_param
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
        self.only_supervised = True
        self.model = Classification(self.model, self.param["output_dim"], self.param["classification_dim"])
        model = self.train()
        self.only_supervised = False
        return model

    def train_epoch(self):
        """ Normal training method used, if we want to predict multiple classes at once.
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

            #TODO aggregate loss
            batch_loss, output = self.train_batch(sentences, target)
            loss += batch_loss
            aggregated_targets.append(target.to(torch.device('cpu')))
            aggregated_outputs.append(output.to(torch.device('cpu')))
        aggregated_targets = torch.cat(aggregated_targets)
        aggregated_outputs = torch.cat(aggregated_outputs)
        if self.param["with_supervised"] or self.only_supervised:
            metrics = calculate_metrics(aggregated_targets, aggregated_outputs, 
                                        micro_average=self.param["use_micro_average"])
            log("Train", metrics)
        return loss

    def train_bs_epoch(self):
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
            # Combining the sample depends on whether we used padding or not (=tensor vs list output of dataloader)
            if isinstance(sentences, list):
                sentences += other_sentences
            else:
                sentences_max_length = sentences.size()[1]
                other_sentences_max_length = other_sentences.size()[1]
                # We need to check which tensor needs additional padding before we can concatenate them
                if sentences_max_length > other_sentences_max_length:
                    new_size = other_sentences.size()[0], sentences_max_length, other_sentences.size()[2]
                    new_other = torch.zeros(new_size)
                    new_other[:,:other_sentences_max_length,:] = other_sentences
                    other_sentences = new_other
                elif sentences_max_length < other_sentences_max_length:
                    new_size = sentences.size()[0], other_sentences_max_length, sentences.size()[2]
                    new_sentences = torch.zeros(new_size)
                    new_sentences[:,:sentences_max_length,:] = sentences
                    sentences = new_sentences
                sentences = torch.cat([sentences, other_sentences])
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
        if self.param["with_supervised"] or self.only_supervised:
            metrics = calculate_metrics(aggregated_targets, aggregated_outputs, 
                                        micro_average=self.param["use_micro_average"])
            log("Train", metrics)
        return loss

    def train_batch(self, sentences, target):
        """ Training method for one data batch.
        """
        self.optimizer.zero_grad()
            
        if self.param["use_padding"]:
            output = self.model(sentences)
        else:
            output = []
            for sentence in sentences:
                output.append(self.model(torch.unsqueeze(sentence, dim=0)))
            output = torch.cat(output, dim=0)
        
        if not self.only_supervised:
            similarity = Class2Simi(target)
            loss = self.learner_clustering.calculate_criterion(output, similarity)
            if self.param["with_supervised"]:
                loss += self.learner_classification.calculate_criterion(output, target)
        else:
            loss = self.learner_classification.calculate_criterion(output, target)

        loss.backward()
        self.optimizer.step()
        return loss, output

    def eval_epoch(self):
        """ Evaluation of one epoch. """
        aggregated_targets = []
        aggregated_outputs = []
        for sentences, entities, attributes in self.validloader:
            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()
            
            loss, output = self.eval_batch(sentences, target)
            aggregated_targets.append(target.to(torch.device('cpu')))
            aggregated_outputs.append(output.to(torch.device('cpu')))
        aggregated_targets = torch.cat(aggregated_targets)
        aggregated_outputs = torch.cat(aggregated_outputs)
        if self.param["with_supervised"] or self.only_supervised:
            metrics = calculate_metrics(aggregated_targets, aggregated_outputs, 
                                        micro_average=self.param["use_micro_average"])
            log("Eval", metrics)
        return loss

    def eval_batch(self, sentences, target):
        """ Evaluation of one batch. """
        if self.param["use_padding"]:
            output = self.model(sentences)
        else:
            output = []
            for sentence in sentences:
                output.append(self.model(torch.unsqueeze(sentence, dim=0)))
            output = torch.cat(output, dim=0)
        
        if not self.only_supervised:
            similarity = Class2Simi(target)
            loss = self.learner_clustering.calculate_criterion(output, similarity)
            if self.param["with_supervised"]:
                loss += self.learner_classification.calculate_criterion(output, target)
        else:
            loss = self.learner_classification.calculate_criterion(output, target)
        return loss, output

    def eval_bs_epoch(self):
        aggregated_targets = []
        aggregated_outputs = []
        for sentences, entities, attributes in self.validloader:
            # After getting a batch from the other classes, simply append them to the current
            # sentences. The error calculation is robust enough.
            other_sentences, other_entities, other_attributes = self.getOtherBatch(train=False)
            # Combining the sample depends on wheter we used padding or not (=tensor vs list output of dataloader)
            if isinstance(sentences, list):
                sentences += other_sentences
            else:
                sentences_max_length = sentences.size()[1]
                other_sentences_max_length = other_sentences.size()[1]
                # We need to check which tensor needs additional padding before we can concatenate them
                if sentences_max_length > other_sentences_max_length:
                    new_size = other_sentences.size()
                    new_size[1] = sentences_max_length
                    new_other = torch.zeros(new_size)
                    new_other[:,:other_sentences_max_length,:] = other_sentences
                    other_sentences = new_other
                elif sentences_max_length < other_sentences_max_length:
                    new_size = sentences.size()
                    new_size[1] = other_sentences_max_length
                    new_sentences = torch.zeros(new_size)
                    new_sentences[:,:sentences_max_length,:] = sentences
                    sentences = new_sentences
                sentences = torch.cat([sentences, other_sentences])
            entities = torch.cat([entities, other_entities])
            attributes = torch.cat([attributes, other_attributes])

            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()

            loss, output = self.eval_batch(sentences, target)
            aggregated_targets.append(target.to(torch.device('cpu')))
            aggregated_outputs.append(output.to(torch.device('cpu')))
        aggregated_targets = torch.cat(aggregated_targets)
        aggregated_outputs = torch.cat(aggregated_outputs)
        if self.param["with_supervised"] or self.only_supervised:
            metrics = calculate_metrics(aggregated_targets, aggregated_outputs, 
                                        micro_average=self.param["use_micro_average"])
            log("Eval", metrics)
        return loss

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

def calculate_metrics(targets, predictions, micro_average=True):
    """ Calculates common performance metrics. 
    Arguments:
        targets {torch.tensor[N]} -- The target values for a given sample.
        predictions {torch.tensor[N x output_dim]} -- The softmax/sigmoid output for
            that sample
        micro_average {bool} -- whether to use micro or macro averaging
    Returns:
        {dict{f1, recall, precision}}
    """
    # If the output dimension is 1, we used the sigmoid function
    used_sigmoid = predictions.size()[1] == 1
    # We want to compute for each prediction the argmax class -> for sigmoid
    if used_sigmoid:
        max_classes = torch.round(predictions)
    else:
        max_classes = torch.argmax(predictions, dim=1)
    max_classes = max_classes.to(torch.device('cpu')).detach().numpy()
    targets = targets.to(torch.device('cpu')).detach().numpy()
    statistic = {}
    average = "micro" if micro_average else "macro"
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
        {(train: Dataset, validation Dataset)}
    """
    assert (validation_percentage >= 0 and validation_percentage <= 1)
    validation_length = math.ceil(len(dataset) * validation_percentage)
    train_length = len(dataset) - validation_length
    datasetsList = random_split(dataset, [train_length, validation_length])
    return datasetsList[0], datasetsList[1]

def log(*string):
    if verbose_training:
        print(string)

