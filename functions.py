import csv

import nltk
from nltk import corpus, word_tokenize, WordNetLemmatizer
from nltk.stem import PorterStemmer
from langdetect import detect
nltk.download('stopwords')
nltk.download('punkt')
import re

def open_data_file(location):
    rows = []
    with open(location) as file:
        csvreader = csv.reader(file, quoting=csv.QUOTE_NONE,  delimiter='\n')
        header = next(csvreader)
        for row in csvreader:
            if len(row) == 1:
                rows.append(row[0])
    return rows



""" This method splits the data into conversations.
    Note each conversations ends with a  sentence in quotation marks 
    If is_test=True, only the first 100 conversations are considered for convenient debugging"""


def genererate_conversations(rows, is_test):
    conversations = []
    current_con = ""
    count = 0
    if is_test:
        for row in rows:
            if row[0] == '"':
                current_con += ' '
                current_con += row.strip('"')
                conversations.append(current_con)
                current_con = ""
                count += 1
                if count == 10000:
                    return conversations
            else:
                current_con += ' '
                current_con += row

    for row in rows:
        if row[0] == '"':
            current_con += row.strip('"')
            conversations.append(current_con)
            current_con = ""
        else:
            current_con += row
    return conversations


""" This  method filters out  non-english conversations and then 
    runs the preprosses method for each conversation """
def process_conversations(conversations):
    processed = []
    for convo in conversations:
        try :
            language = detect(convo)
        except:
            language = 'error'

        if language == 'en':
            processed.append(preprocess(convo))
    return processed


""" This method processes the text and tokenizes the conversation"""
def preprocess(conversation):
    filtered = punctuation_handles(conversation.lower())
    tokens = tokenize_stem_stop(filtered, True, True)
    return tokens


""" this method filters the text by removing  @xxxxx twitter handles,
 hyperlinks, and special characters (including emojiis) """


def punctuation_handles(con):
    no_handles = re.sub(r'@[1-9a-zA-Z]+', r'', con)
    no_links = re.sub(r'https[^ \t\n]+', r'', no_handles)
    no_specialchar = re.sub(r'[^a-zA-Z ]+', r'', no_links)
    return no_specialchar


def tokenize_stem_stop(text, stem=True, lemma=True):
    tokens = word_tokenize(text)
    stop_words = set(corpus.stopwords.words("english"))
    stop_words.union({'u','hi', 'sorri'})

    filtered_sentence = []

    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    if lemma:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    return filtered_sentence
