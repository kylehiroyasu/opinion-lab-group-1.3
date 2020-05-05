import pandas as pd
from bs4 import BeautifulSoup


def doc_to_jsonl(doc_string):
    """Expects an entire file as a single string and returns the 
    elements deemed important for machine learning task
    CAUTION: duplicate sentences when there are multiple attribute
    entity pairs in a sentence.

    Arguments:
        doc_string {string} -- a single string containing the 
        entire file contents

    Returns:
        list -- a list of simple dictionaries where all values
        of the dictionary are string values. If data was not 
        present in document, then field is ommitted.
    """    
    soup = BeautifulSoup(doc_string, "lxml")
    docs = []
    for idx, sentence in enumerate(soup.find_all('sentence')):
        opinions = sentence.find_all('opinion')

        if len(opinions) == 0:
            doc = {
                'rid': sentence.attrs['id'].split(':')[0],
            }
            doc.update(sentence.attrs)
            doc['text'] = sentence.find('text').contents[0]
            docs.append(doc)

        for opinion in opinions:
            doc = {
                'rid': sentence.attrs['id'].split(':')[0],
                'entity': opinion.attrs['category'].split('#')[0],
                'attribute': opinion.attrs['category'].split('#')[1],
                'polarity': opinion.attrs['polarity']
            }
            doc.update(sentence.attrs)
            doc['text'] = sentence.find('text').contents[0]
            docs.append(doc)
    return docs


def load_data_as_df(path):
    """ Takes a file path and returns the important parts as denormalized
    pandas dataframe

    Arguments:
        path {path or string object} -- path to file

    Returns:
        DataFrame -- denormalized dataframe of all the sentences and 
        entity attribute pairs associated with it. If no pairs were present
        the fields are left blank.
    """    
    with open(path, mode='r') as f:
        doc = f.readlines()
    doc_string = ' '.join(doc)
    list_of_docs = doc_to_jsonl(doc_string)
    df = pd.DataFrame(list_of_docs)
    return df
