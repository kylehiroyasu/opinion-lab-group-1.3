from bs4 import BeautifulSoup
import pandas as pd
import re
import spacy

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

def clean_texts(docs, remove_stopwords=False, lemmatize=False):
    """Removes known typos, use the two named args you can also filter
    stop words and lemmatize tokens if desired.

    Arguments:
        docs {list of dicts} -- function takers iterable and modifies 
        the text value of each document

    Keyword Arguments:
        remove_stopwords {bool} --  remove stopwords (default: {False})
        lemmatize {bool} -- lemmatize tokens (default: {False})

    Returns:
        list of dicts -- returns a list of dicts in the same format as 
        passed in. The only modifications are the documents text values
    """    
    space_regex = re.compile('\xa0')
    love_regex = re.compile('[l]+[o]+[v]+[e]+', re.IGNORECASE)
    wow_regex = re.compile('W-O-W')
    perfect_regex = re.compile('P-E-R-F-E-C-T')
    
    nlp = spacy.load("en_core_web_sm")

    new_docs = []
    for doc in docs
        text = doc['text']
        text = space_regex.replace(' ', text)
        text = love_regex.replace('love', text)
        text = wow_regex.replace('wow', text)
        text = perfect_regex.replace('perfect', text)
        spacy_doc = nlp(t)
        selected_tokens = []
        for token in spacy_doc:
            if token.is_stop and remove_stopwords:
                continue
            if lemmatize:
                selected_tokens.append(token.lemma_)
            else:
                selected_tokens.append(token.text)
        text = ' '.join(selected_tokens)
        doc['text'] = text
        new_docs.append(doc)
    return new_docs

def load_data_as_df(path, remove_stopwords=False, lemmatize=False):
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
    cleaned_docs = clean_texts(list_of_docs, 
                                remove_stopwords=remove_stopwords, 
                                lemmatize=lemmatize)
    df = pd.DataFrame(list_of_docs)
    return df
