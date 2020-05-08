import torch as t
from torch.utils.data import Dataset
from flair.data import Sentence

import pandas as pd

class AspectDataset(Dataset):

    def __init__(self, sentences, entities, entity_dict, attributes, attribute_dict):
        """ Intializes the AspectDataset. Important: For the labels use the same order
        as the sentences, thus the sentences are linked to the labels with the same index
        TODO: What happens with Sentences with multiple labels? Are we working with lists
        of labels or with "multiple" sentences with each one label 
        Arguments: 
            sentences {list of Flair.Sentence} -- Sentences used in this dataset
            entities {list of int} -- the numeric values for the entity labels
            entity_dict {dict(int, String)} -- mapping for the numeric entity values to
            the corresponding name
            attirbutes {list of int} -- numeric values for the attribute labels
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
        self.embeddings = None

    def addEmbeddings(self, embeddings):
        """ Adds embeddings to the dataset. If you do this the items will be embeddings
        instead of Sentence objects.
        Arguments:
            embeddings (flair.WordEmbeddings) -- used to convert the sentence to an embedding
        """
        self.embeddings = embeddings

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
            return (self.sentences[idx], self.entities[idx], self.attributes[idx])
        else:
            sentence = self.embeddings.embed(self.sentences[idx])
            sentence = [t.unsqueeze(token.embedding, dim=0) for token in sentence]
            sentence = t.cat(sentence, dim=0)
            return (sentence, self.entities[idx], self.attributes[idx])


def dfToDataset(df, entity_dict, attribute_dict):
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
    Returns:
        {AspectDataset} -- extracted dataset from the 
    """
    sentences = []
    entities = []
    attributes = []
    for row in df.itertuples():
        if pd.notna(row.attribute) and pd.notna(row.entity):
            sentences.append(Sentence(row.text, use_tokenizer=True))
            entities.append(entity_dict[row.entity])
            attributes.append(attribute_dict[row.attribute])

    entity_id_dict = {}
    attribute_id_dict = {}
    for key, value in entity_dict.items():
        entity_id_dict[value] = key
    for key, value in attribute_dict.items():
        attribute_id_dict[value] = key
    return AspectDataset(sentences, entities, entity_id_dict, attributes, attribute_id_dict)
