from torch.utils.data import DataLoader, random_split

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
        for sentences, entities, attributes in self.dataloader:
            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()

            self.optimizer.zero_grad()
            
            if param["use_padding"]:
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
        for sentences, entities, attributes in self.validloader:
            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()
            
            if param["use_padding"]:
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

    def train_bs_epoch(self):
        for sentences, entities, attributes in self.dataloader:
            other_sentences, other_entities, other_attributes = self.getOtherBatch(train=True)
            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()

            self.optimizer.zero_grad()
            
            if param["use_padding"]:
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

    def eval_bs_epoch(self):
        for sentences, entities, attributes in self.validloader:
            other_sentences, other_entities, other_attributes = self.getOtherBatch(train=False)
            if self.param["train_entities"]:
                target = entities
            else:
                target = attributes
            if self.param["cuda"]:
                sentences = sentences.cuda()
                target = target.cuda()
            
            if param["use_padding"]:
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

    def getOtherBatch(train)
        if train:
            dataloader = self.other_dataloader
        else:
            dataloader = self.other_validloader
        for data in dataloader:
            return data


def split_datset(dataset, validation_percentage):
    validation_length = math.ceil(len(dataset) * validation_percentage)
    train_length = len(train_dataset) - validation_length
    datasetsList = random_split(train_dataset, [train_length, validation_length])
    return datasetsList[0], datasetsList[1]

def log(*string):
    if verbose_training:
        print(string_list)

