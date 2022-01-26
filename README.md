# CiphixCase

This repository implements my solution to the  case that was provided to me by Ciphix. 
_The goal of the code is to extract the  top 10 topics/ types of questions that are present in the data._
To this end a the NLTK library is used, as well as several machine learning methods.

Example conversation:
- _@sprintcare is the worst customer service_
-  _"@115713 This is saddening to hear. Please shoot us a DM, so that we can look into this for you. -KC"_
# Exploratory Analysis and Preprocessing
The data consists of approximately 1.1 million  customer conversations. The last sentence of a conversation is in 
quotation marks, which we will use to split the data. The main (and target) language is english,  although some Spanish, 
Chinese and Arabic is also present. To ensure correct results, we will filter these out. 
Every sentence begins with a @xxxxx Twitter handle, these will be removed as they are not relevant
for the topic of the conversation. Furthermore, we will remove all special characters, emojis and stopwords. 
Next we tokenized our conversations as well as stemmed and lemmatized the tokens. This  leaves us  with several keywords per  
conversation that (hopefully) are relevant to the topic.
The preprocessing methods have been implemented as multicore per default, a single thread option has also been included, however
to ensure compatebility with all machines. 

As a final step of the preprocessing phase the conversations were represented in a numerical way using SciKit learn's TfidfVectorizer.

# Methods Used
Two machine learning methods were used. The first attempt was done using 
the K-Means Clustering algorithm. Then for each  of the  clusters, we got the top 10 words based on TF-IDF score. 
This did not yield satisfactory results. Next we made use of  Latent Dirichlet allocation (LDA) to perform the 
Topic Extraction. We were left with  the following 10 topics:

| Topics                      | Keywords                        |
|-----------------------------|---------------------------------|
| Transportation / Rail       | Train, Service, Delay, Ticket   |
| Technical Support / General | Issue, Update                   |
| Support Request             | Send, Email, Help, Assist       |
| Gaming / Software Support   | Game, App, Play, Error          |
| Technical Support / Telecom | Iphone, Phone, Update           |
| Retail                      | Store, Card, Return             |
| Telecom                     | Account, Phone, Number, Service |
| Delivery / Logistics        | Order, Package, Delivery        |
| Transportation / Airline    | Flight, Time, Delay             |
| General Support             | Email, Account, Adress, Private |

# Discussion
Due to its unsupervised nature, topic discovery is a tough problem.
There are however some methods other than LDA that can be used to solve it.
One such method is Latent Semantic Analysis. It constructs a word occurrence matrix,
which is then analized to find the patterns with the text. 

# User Guide
The code in this repository can be  used as follows:
1. Load the  file and split it into conversations (Check for correct separator)
2. Pre-Process the conversations using `process_conversations()` Note that this is multicore par default
3. For cluster based analysis call `cluster_conversations()`
4. For LDA analysis call `LDA_analysis()`