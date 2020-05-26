"""
Taken and apdated from https://github.com/GT-RIPL/L2C
The classes are heavily edited and basically only their idea
is kept here. Their function is to automatically calculate 
the loss of a given NN, its outputs and class labels.
The whole training loop will be done externally.
"""
import torch as t
import torch.nn as nn
from Loss import PairEnum

class Learner_Clustering():

    def __init__(self, criterion):
        self.criterion = criterion

    def calculate_criterion(self, output, similarity, mask=None):
        """ Calculates the loss of the NN.
        Arguments:
            output {torch.tensor[N x output_dim]} -- the outputs from
            the last layer of the NN
            similarity {torch.tensor[N x N]} -- the similarity value for each
            of the outpus (generated from Class2Simi)
            mask {torch.tensor[N]} -- mask to exclude certain samples
        Returns:
            loss: torch.tensor[1]
        """
        prob1, prob2 = PairEnum(output, mask)
        return self.criterion(prob1, prob2, similarity)
