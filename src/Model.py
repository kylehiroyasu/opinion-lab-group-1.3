import torch as t
import torch.nn as nn

class Model(nn.Module):

    def __init__(self, embedding_dim, output_dim):
        """ Initilizes the Model. Creates the attention matrix M as well
        as the linear layer used afterwards. For more information see the
        ABAE paper.

        Arguments:
            embedding_dim {int} -- variable for saving the size of the embeddings, 
                used as the size for the attention matrix M as well
            output_dim {int} -- output dimension of the models last layer (softmax)
        """
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.attention_matrix = t.rand((self.embedding_dim, self.embedding_dim), requires_grad=True)
        self.linear = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        """Expects a batch of sentences and produces the softmax scores usable for
        Supervised Learning or the KCL/MCL objectives.

        Arguments:
            x {tensor [batch size, sentence length, embedding_dim]} -- tensor containing
                one batch of data

        Returns:
            tensor [batch size, output_dim] -- softmax scores for each sentence
                in the batch
        """   
        average = self.average(x)
        attention_logit = self.attention_logit(average, self.attention_matrix)
        d = self.attention_d(x, attention_logit)
        attention_weight = self.softmax(d)
        sentence_embedding = self.weighted_sum(attention_weight, x)
        z = self.linear(sentence_embedding)
        return self.softmax(z)


    def average(self, x):
        return 1/(x.size()[1]) * t.sum(x, dim=1)

    def attention_logit(self, avg, attention_matrix):
        return t.matmul(attention_matrix, avg.T).T

    def attention_d(self, x, logit):
        d =  t.matmul(x, logit[:,:,None])
        return d.reshape(d.size()[0], -1)

    def softmax(self, d):
        # Max returns a tuple with the [values, indices]
        # Subtract the maximum for a numerically stable softmax
        z = t.exp(d - t.max(d, dim=1)[0][:,None])
        softmax = z/(t.sum(z, dim=1)[:,None])
        return softmax

    def weighted_sum(self, weights, x):
        return t.sum(x * weights[:,:, None], dim=1)

class Classification(nn.Module):

    def __init__(self, previous_model, input_dim, output_dim):
        """ Initializes a classification layer. This can be used to 
        identify which class belongs to which output of the previous
        model. In a binary case this might not be necessary, but it
        might be beneficial to use multidimensional inputs for better
        coverage of the clusters. This module is used for a mapping
        and should be trained after the other model finished training.
        The non-linearity used is based on the output dimension:
        1 = Sigmoid, 1 < n = Softmax
        Arguments:
            previous_model {Model} -- the previously trained model
            input_dim {int} -- Size of the output of the previous model
            output_dim {int} -- Number of classes this model should 
                predict
        """
        super(Classification, self).__init__()
        self.model = previous_model
        self.linear = nn.Linear(input_dim, output_dim)
        if  output_dim > 1:
            self.output = nn.Softmax()
        else:
            self.output = nn.Sigmoid()

    def forward(self, x):
        """Expects a batch of embedded sentences and produces the class
            prediction for each of them.

        Arguments:
            x {tensor [batch size, input_dim]} -- tensor containing
            one batch of data

        Returns:
            tensor [batch size, output_dim] -- classification scores 
                for each sentence in the batch
        """ 
        x = self.model(x)
        x = self.linear(x)
        return self.output(x)
