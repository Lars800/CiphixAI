import csv
from gensim import corpora, models
import gensim
import functions
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

if __name__ == "__main__":
    rows = []
    with open('data.csv') as file:
        csvreader = csv.reader(file, quoting=csv.QUOTE_NONE,  delimiter='\n')
        header = next(csvreader)
        for row in csvreader:
            if len(row) == 1:
                rows.append(row[0])

    #split and process data
    conversations = functions.genererate_conversations(rows, False)
    processed = functions.process_conversations(conversations)

    #generate dictionary from tokens
    dictionary = gensim.corpora.Dictionary(processed)
    dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)
    bagged_convos = [dictionary.doc2bow(convo) for convo in processed]

    #run Latent Dirichlet allocation model
    lda_model = gensim.models.LdaMulticore(bagged_convos, num_topics=10, id2word=dictionary, passes=2, workers=2)
    for idx, topic in lda_model.print_topics(-1):
        print('Topic: {} \nWords: {}'.format(idx, topic))




