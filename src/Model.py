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
            output_dim {int} -- output dimension of the models last layer (softmax/sigmoid)
        """
        super(Model, self).__init__()
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.attention_matrix = t.rand((self.embedding_dim, self.embedding_dim), requires_grad=True)
        self.att_linear = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.linear = nn.Linear(self.embedding_dim, self.output_dim)
        self.use_softmax = True

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
        if self.use_softmax:
            if self.output_dim == 1:
                return self.sigmoid(z)
            return self.softmax(z)
        else:
            return z

    def to(self, device):
        super(Model, self).to(device)
        self.attention_matrix = self.attention_matrix.to(device)
        
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

    def sigmoid(self, d):
        return 1/(1+t.exp(-d))

    def weighted_sum(self, weights, x):
        return t.sum(x * weights[:,:, None], dim=1)

# Taken and adapted from Group 1.2. for comparison
class LinModel(nn.Module):

    def __init__(self, word_dim, output_dim):
        super().__init__()

        self.attention = nn.Linear(word_dim, 1, bias=False)
        self.classifier = nn.Linear(word_dim, output_dim)
        self.use_softmax = True
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        shape_len = len(x.shape)

        # sum up word vectors weighted by their word-wise attention
        attentions = self.attention(x)
        x =  attentions * x
        if shape_len == 3:
            x = x.sum(axis=1)
        elif shape_len == 2:
            x = x.sum(axis=0)

        # feed it to the classifier
        x = self.classifier(x)

        # apply softmax
        if self.use_softmax:
            x = self.softmax(x)

        return x

class Classification(nn.Module):

    def __init__(self, previous_model, input_dim, output_dim=1):
        """ Initializes a classification layer. This can be used to 
        identify which class belongs to which output of the previous
        model. In a binary case this might not be necessary, but it
        might be beneficial to use multidimensional inputs for better
        coverage of the clusters. This module is used for a mapping
        and should be trained after the other model finished training.
        Arguments:
            previous_model {Model} -- the previously trained model
            input_dim {int} -- Size of the output of the previous model
            output_dim {int} -- Number of classes this model should 
                predict
        """
        super(Classification, self).__init__()
        self.model = previous_model
        self.model.use_softmax = True
        self.relu = nn.ReLU()
        self.linear = nn.Linear(input_dim, output_dim)
        if output_dim == 1:
            self.sigmoid = nn.Sigmoid()
        else:
            self.sigmoid = nn.Softmax(dim=1)

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
        if not self.model.use_softmax:
            x = self.relu(x)
        x = self.linear(x)
        x = self.sigmoid(x)
        return x

def save_model(model, path):
    t.save(model.state_dict(), path)
        
