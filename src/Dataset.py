import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from flair.data import Sentence

import pandas as pd

class AspectDataset(Dataset):

    def __init__(self, sentences, entities, entity_dict, attributes, attribute_dict, embeddings=None, ):
        """ Intializes the AspectDataset. Important: For the labels use the same order
        as the sentences, thus the sentences are linked to the labels with the same index
        TODO: What happens with Sentences with multiple labels? Are we working with lists
        of labels or with "multiple" sentences with each one label 
        Arguments: 
            sentences {list of Flair.Sentence} -- Sentences used in this dataset
            entities {list of Tensor[]} -- the numeric values for the entity labels
            entity_dict {dict(int, String)} -- mapping for the numeric entity values to
            the corresponding name
            attirbutes {list of Tensor[]} -- numeric values for the attribute labels
            attribute_dict {dict(int, String)} -- mapping for the numeric attribute values
            to the corresponding name
        """
        assert(len(sentences) == len(entities))
        assert(len(sentences) == len(attributes))
        self.sentences = sentences
        self.entities = entities
        self.entity_dict = entity_dict
        self.attributes = attributes
        self.attribute_dict = attribute_dict
        self.embeddings = embeddings

    def addEmbeddings(self, embeddings):
        """ Adds embeddings to the dataset. If you do this the items will be embeddings
        instead of Sentence objects.
        Arguments:
            embeddings (flair.WordEmbeddings) -- used to convert the sentence to an embedding
        """
        self.embeddings = embeddings

    def targetLengths(self):
        """ Returns the length of the entity_dict and attribute_dict """
        entity_length = len(self.entity_dict) if len(self.entity_dict) > 0 else 1
        attribute_length = len(self.attribute_dict) if len(self.attribute_dict) > 0 else 1
        return entity_length, attribute_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        """ Returns the sample at the specific index. If embeddings were added to
        this dataset, this method returns the embeddings instead of the Sentence
        object.
        Arguments:
            idx {int} -- Index for the wanted sample
        Returns:
            (Sentence, int: entity, int: attribute)
        """
        if self.embeddings is None:
            raise RuntimeError("No embeddings defined for the dataset.")
        else:
            sentence = self.sentences[idx]
            self.embeddings.embed(sentence)
            sentence = [torch.unsqueeze(token.embedding, dim=0) for token in sentence]
            sentence = torch.cat(sentence, dim=0)
            return (sentence, self.entities[idx], self.attributes[idx])

def dfToBinarySamplingDatasets(df, use_attributes, target_class, embeddings=None):
    """ Expects as pandas.DataFrame with the columns: text, entity, attribute.
    Creates an AspectDataset. Dicts are used to map the numeric values to their
    names. If there are still rows with NaN values for the labels, they will be
    filtered here. Uses the standard tokenizer.
    Arguments:
        df {pandas.DataFrame} -- Dataframe with the sentences and their aspects
        use_attributes {bool} -- Whether to split the dataset on attributes or 
            enitites
        target_class {String} -- The wanted target to sample from. Wheter it
            is an entitiy or attribute is denoted by use_attributes
        embeddings {Flair.Embedding} -- Embeddings used in the Dataset
    Returns:
        {targetDataset: AspectDataset, otherDataset: AspectDataset} -- extracted 
            datasets from the Dataframe. targetDataset only has samples form the
            target_class. The other samples are all in otherDataset 
    """
    target_sentences = []
    other_sentences = []
    for row in df.itertuples():
        if use_attributes:
            row_class = row.attribute
        else:
            row_class = row.entity
        if pd.notna(row_class):
            if row_class == target_class:
                target_sentences.append(Sentence(row.text, use_tokenizer=True))
            else:
                other_sentences.append(Sentence(row.text, use_tokenizer=True))
        else:
            if target_class != "NaN":
                other_sentences.append(Sentence(row.text, use_tokenizer=True))
            else:
                target_sentences.append(Sentence(row.text, use_tokenizer=True))
    target_class = [torch.tensor([1.]) for _ in range(len(target_sentences))]
    other_class = [torch.tensor([0.]) for _ in range(len(other_sentences))]
    targetDataset = AspectDataset(target_sentences, target_class, {}, target_class, {}, embeddings)
    otherDataset = AspectDataset(other_sentences, other_class, {}, other_class, {}, embeddings)
    return targetDataset, otherDataset

def dfToDataset(df, entity_dict, attribute_dict, embeddings=None):
    """ Expects as pandas.DataFrame with the columns: text, entity, attribute.
    Creates an AspectDataset. Dicts are used to map the numeric values to their
    names. If there are still rows with NaN values for the labels, they will be
    filtered here. Uses the standard tokenizer.
    Arguments:
        df {pandas.DataFrame} -- Dataframe with the sentences and their aspects
        entity_dict {dict(String, int)} -- mapping for entity names to their
            numeric entity values
        attribute_dict {dict(String, int)} -- mapping for attribute names to
            their numeric values 
        embeddings {Flair.Embedding} -- Embeddings used in the Dataset
    Returns:
        {AspectDataset} -- extracted dataset from the 
    """
    sentences = []
    entities = []
    attributes = []
    for row in df.itertuples():
        if pd.notna(row.attribute) and pd.notna(row.entity):
            sentences.append(Sentence(row.text, use_tokenizer=True))
            entities.append(torch.tensor([entity_dict[row.entity]]))
            attributes.append(torch.tensor([attribute_dict[row.attribute]]))
        else: 
            sentences.append(Sentence(row.text, use_tokenizer=True))
            entities.append(torch.tensor([entity_dict["NaN"]]))
            attributes.append(torch.tensor([attribute_dict["NaN"]]))

    entity_id_dict = {}
    attribute_id_dict = {}
    for key, value in entity_dict.items():
        entity_id_dict[value] = key
    for key, value in attribute_dict.items():
        attribute_id_dict[value] = key
    return AspectDataset(sentences, entities, entity_id_dict, attributes, attribute_id_dict, embeddings)


def collate(batch):
    sentences = [sample[0] for sample in batch]
    entities = torch.cat([sample[1] for sample in batch])
    attributes = torch.cat([sample[2] for sample in batch])
    return sentences, entities, attributes

def collate_padding(batch):
    sentences = pad_sequence([sample[0] for sample in batch], batch_first=True)
    entities = torch.cat([sample[1] for sample in batch])
    attributes = torch.cat([sample[2] for sample in batch])
    return sentences, entities, attributes
