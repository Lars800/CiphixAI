from multiprocessing import cpu_count
from gensim import corpora, models
import gensim
import functions
import nltk
import os


if __name__ == "__main__":
    location = os.path.split(os.getcwd())[0] + '/data.csv'

    # open, split and process data
    data = functions.open_data_file(location)
    conversations = functions.genererate_conversations(data, is_test=False)
    processed = functions.process_conversations(conversations)
