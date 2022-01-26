import csv
import collections
import math

import nltk
import numpy as np
from nltk import corpus, word_tokenize, WordNetLemmatizer
from nltk.stem import PorterStemmer
from langdetect import detect
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import gensim
nltk.download('stopwords')
nltk.download('punkt')
import re
from multiprocessing import Pool, cpu_count


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
                if count == 100:
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

"""  This method  passes the conversations to to the processing method,
    Per default it does so in a parralel way    """
def process_conversations(conversations, multi_core=True):
    if multi_core:
        n_threads = cpu_count() - 1
        chunks = math.floor(len(conversations) /  n_threads)
        with Pool(n_threads) as pool:
            result = pool.map(func=preprocess, iterable=conversations,
                              chunksize=chunks)
            pool.close()
            pool.join()
        for convo in result:
            if convo  ==  -1:
                result.remove(convo)

        return result

    else:
        processed = []
        for convo in conversations:
            result = preprocess(convo)
            if result != -1:
                processed.append(result)
        return processed

""" This method filters, processes and tokenizes the conversation"""
def preprocess(conversation):
    try:
        language = detect(conversation)
    except:
        language = 'error'

    if language == 'en':
        filtered = punctuation_handles(conversation.lower())
        tokens = tokenize_stem_stop(filtered, True, True)
        return tokens
    else:
        return -1


""" this method filters the text by removing  @xxxxx twitter handles,
 hyperlinks, and special characters (including emojiis) """
def punctuation_handles(con):
    no_handles = re.sub(r'@[1-9a-zA-Z]+', r'', con)
    no_links = re.sub(r'https[^ \t\n]+', r'', no_handles)
    no_specialchar = re.sub(r'[^a-zA-Z ]+', r'', no_links)
    return no_specialchar

""" This method  tokenizes and  processes the words"""
def tokenize_stem_stop(text, stem=True, lemma=True):
    tokens = word_tokenize(text)
    stop_words = set(corpus.stopwords.words("english"))
    stop_words.union({'u','hi', 'thi', 'thank', 'plea', 'sorri', 'dm'})

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

""" This method implements TF-IDF vectorization"""
def vectorize_conversations(token_list, min=0.1, max=0.7, return_top_words=False):
    joined_tokens = []
    for token in token_list:
        tokens_raw = " ".join(token)
        joined_tokens.append(tokens_raw)

    vectorizer = TfidfVectorizer(max_df=max,
                                 min_df=min,
                                 lowercase=True)
    response = vectorizer.fit_transform(joined_tokens)

    if return_top_words:
        feature_array = np.array(vectorizer.get_feature_names())
        tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]
        n = 10
        return feature_array[tfidf_sorting][:n]

    return response

""" This method clusters the conversations, and extracts the topics per cluster"""
def cluster_conversations(conversations, clusters):
    vector_model = vectorize_conversations(conversations)
    km_model = KMeans(n_clusters=clusters)
    km_model.fit(vector_model)
    clustering = collections.defaultdict(list)
    for idx, label in enumerate(km_model.labels_):
        clustering[label].append(idx)

    topics = []
    for cluster in clustering:
        current_obs = []
        for obs in clustering[cluster]:
            current_obs.append(conversations[obs])
        topics.append(vectorize_conversations(current_obs, min=2, max=0.6, return_top_words=True))

    return topics

