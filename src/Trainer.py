from torch.utils.data import DataLoader, random_split
import numpy as np

from Model import Model, Classification
from Dataset import collate, collate_padding
from Loss import *

param = {
    "embedding_dim": hidden_dim,
    "output_dim": output_dim,
    "epochs": 40,
    "lr": 0.0025,
    "batch_size": 256,
    "use_padding": False,
    "validation_percentage": 0.1,
    "binary_sampling_percentage": 0.5,
    "cuda": False,
    "use_kcl": True,
    "with_supervised": False,
    "train_entities": True
}

verbose_training = True

class Trainer:

    def __init__(dataset, param_dict, other_dataset=None):
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
        self.use_train_iterator = None

    def train(verbose=True):
        """ Starts the training procedure.
        Arguments:
            verbose {bool} -- Wheter to log log messages during training
        Returns:
            {Model} -- The trained model
        """

        verbose_training = verbose

        train_dataset, validation_dataset = split_datset(self.dataset, param["validation_percentage"])
        if self.binary_sampling:
            other_train_dataset, other_val_dataset = split_dataset(self.other_dataset, param["validation_percentage"])

        if param["use_padding"]:
            collate_fn = collate_padding
        else:
            collate_fn = collate

        self.dataloader = DataLoader(train_dataset, batch_size=self.param["batch_size"], shuffle=True, collate_fn=collate_fn)
        self.validloader = DataLoader(validation_dataset, batch_size=self.param["batch_size"], collate_fn=collate_fn)
        if self.binary_sampling:
            batch_size = int(round(self.param["batch_size"]*self.param["binary_sampling_percentage"]))
            self.other_dataloader = DataLoader(other_train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
            self.other_validloader = DataLoader(other_val_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

        self.optimizer = t.optim.Adam(self.model.parameters(), lr=self.param["lr"])
        self.learner_classification = Learner_Classification(nn.CrossEntropyLoss())
        if self.param["use_kcl"]:
            self.learner_clustering = Learner_Clustering(KCL())
        else:
            self.learner_clustering = Learner_Clustering(MCL())

        if self.param["cuda"] and t.cuda.is_available():
            log("Using GPU")
            device = t.device('cuda:0')
            self.model.to(device)
        else:
            log("Using CPU")
            self.param["cuda"] = False
            device = t.device('cpu')

        agg_cost = None
        for e in range(self.param["epochs"]):
            log("Epoch:", e)
            self.model.train()
            if self.binary_sampling:
                loss = self.train_bs_epoch()
            else:
                loss = self.train_epoch()
            log("Train loss:", loss)
            if agg_cost is None:
                agg_cost = loss
            else:
                agg_cost += loss
            log("Agg Loss:", loss)

            self.model.eval()
            if self.binary_sampling:
                eval_loss = self.eval_bs_epoch()
            else:
                eval_loss = self.eval_epoch()
            log("Eval Loss:", eval_loss)
            # TODO How do we do the evaluation, if we are not in the supervised case? Assign output to majority label? Compute centroids?

    def train_epoch(self):
        """ Normal training method used, if we want to predict multiple classes at once.
        The parameters for training are all stored in self.param
        """
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
            loss += self.train_batch(sentences, target)
        return loss

    def train_bs_epoch(self):
        """ Training method, if we want to only predict one class -> transformation of
        the problem to binary classification.
        The parameters for training are all stored in self.param
        """
        loss = torch.zeros(1)
        for sentences, entities, attributes in self.dataloader:
            # After getting a batch from the other classes, simply append them to the current
            # sentences. The error calculation is robust enough.
            other_sentences, other_entities, other_attributes = self.getOtherBatch(train=True)
            sentences = torch.cat([sentences, other_sentences], dim=0)
            entities = torch.cat([entities, other_entities], dim=0)
            attributes = torch.cat([attributes, other_attributes], dim=0)

            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()

            loss += self.train_batch(sentences, target)
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
                output.append(self.model(t.unsqueeze(sentence, dim=0)))
            output = t.cat(output, dim=0)

        similarity = Class2Simi(target)
        loss = self.learner_clustering.calculate_criterion(output, similarity)
        if self.param["with_supervised"]:
            loss += self.learner_classification.calculate_criterion(output, target)

        loss.backward()
        self.optimizer.step()
        return loss

    def eval_epoch(self):
        """ Evaluation of one epoch. """
        for sentences, entities, attributes in self.validloader:
            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()
            
            loss = self.eval_batch(sentences, target)
        return loss

    def eval_batch(self, sentences, target):
        """ Evaluation of one batch. """
        if self.param["use_padding"]:
            output = self.model(sentences)
        else:
            output = []
            for sentence in sentences:
                output.append(self.model(t.unsqueeze(sentence, dim=0)))
            output = t.cat(output, dim=0)

        similarity = Class2Simi(target)
        loss = self.learner_clustering.calculate_criterion(output, similarity)
        if self.param["with_supervised"]:
            loss += self.learner_classification.calculate_criterion(output, target)
        return loss

    def eval_bs_epoch(self):
        for sentences, entities, attributes in self.validloader:
            # After getting a batch from the other classes, simply append them to the current
            # sentences. The error calculation is robust enough.
            other_sentences, other_entities, other_attributes = self.getOtherBatch(train=False)
            sentences = torch.cat([sentences, other_sentences], dim=0)
            entities = torch.cat([entities, other_entities], dim=0)
            attributes = torch.cat([attributes, other_attributes], dim=0)

            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()
            
            #TODO aggregate the errors
            loss = self.eval_batch(sentences, target)
        return loss

    def getOtherBatch(self, train):
        """ This method basically gives you the next batch of samples from the other classes
        in the binary classificationi case. It is saving the state of a iterator and if 
        that iterator is done, gets a new one. The datasets must be non-empty. """
        # We already have an iterator and the iterator is for the same dataset.
        if self.use_train_iterator is not None and self.use_train_iterator == train:
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

    def calculate_metrics(targets, predictions, min_target=0, max_target=1, micro_average=True):
        """ Calculates common performance metrics. 
        Arguments:
            targets {torch.tensor[N]} -- The target values for a given sample.
            predictions {torch.tensor[N x output_dim]} -- The softmax/sigmoid output for
                that sample
            min_target {int} -- the minimal possible class target
            max_target {int} -- the maximal possible class target
            micro_average {bool} -- whether to use micro or macro averaging
        Returns:
            {dict{f1, recall, precision}}
        """
        # If the output dimension is 1, we used the sigmoid function
        used_sigmoid = prediction.size()[1] == 1
        # We want to compute for each prediction the argmax class -> for sigmoid
        if used_sigmoid:
            max_classes = torch.round(predictions)
        else:
            max_classes = torch.argmax(predictions, dim=1)
        counts = {}
        for i in range(min_target, max_target+1):
            counts[i] = {"TP":0, "FP":0, "TN": 0, "FN": 0}
        max_classes = max_classes.to(torch.device('cpu')).detach().numpy()
        targets = targets.to(torch.device('cpu')).detach().numpy()
        # We want to go over each possible class in the sample and compute
        # its individual true pos, true neg, false pos and false neg counts
        for i in range(min_target, max_target+1):
            counts[i]["TP"] += np.sum((targets == max_classes) & (targets == i))
            counts[i]["FP"] += np.sum((max_classes == i) & (targets != i))
            #TODO Does this really work like this with multiple targets?
            counts[i]["TN"] += np.sum((max_classes != i) & (targets != i))
            counts[i]["FN"] += np.sum((max_classes != i) & (targets == i))
        statistic = {}
        if micro_average:
            statistic["precision"] = sum([counts[i]["TP"] for i in range(min_target, max_target+1)]) /
                                    sum([counts[i]["TP"]+counts[i]["FP"] for i in range(min_target, max_target+1)])
            statistic["recall"] =  sum([counts[i]["TP"] for i in range(min_target, max_target+1)]) /
                                    sum([counts[i]["TP"]+counts[i]["FN"] for i in range(min_target, max_target+1)])
            statistic["f1"] = 2 * (statistic["precision"] * statistic["recall"])/(statistic["precision"] + statistic["recall"])
        else:
            statistic_per_class = {}
            for i in range(min_target, max_target+1):
                statistic_per_class[i] = {}
                statistic_per_class[i]["precision"] = counts[i]["TP"] / (counts[i]["TP"] + counts[i]["FP"])
                statistic_per_class[i]["recall"] = counts[i]["TP"] / (counts[i]["TP"] + counts[i]["FN"])
                statistic_per_class[i]["f1"] = 2 * (statistic_per_class[i]["precision"] * statistic_per_class[i]["recall"])/
                                                (statistic_per_class[i]["precision"] + statistic_per_class[i]["recall"])
            for stat in ["precision", "recall", "f1"]:
                statistic[stat] = sum([statistic_per_class[i][stat] for i in range(min_target, max_target+1)]) / (max_target-min_target+1)
        return statistic
        

def split_datset(dataset, validation_percentage):
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
    train_length = len(train_dataset) - validation_length
    datasetsList = random_split(train_dataset, [train_length, validation_length])
    return datasetsList[0], datasetsList[1]

def log(*string):
    if verbose_training:
        print(string)

