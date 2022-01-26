# CiphixCase

This repository implements my solution to the  case that was provided to me by Ciphix. 
_The goal of the code is to extract the  top 10 topics/ types of questions that are present in the data._
To this end a the NLTK library is used, as well as several machine learning methods.

# Exploratory Analysis and Preprocessing
The data consists of approximately 1.1 million  customer conversations. The final sentence of a conversation is in 
quotation marks, which we will use to split the data. The main (and target) language is english,  although some Spanish, 
Chinese and Arabic is also present. To ensure correct results, we will filter these out. 
Every sentence begins with an @xxxxx Twitter handle, these will be removed as they are not relevant
for the topic of the conversation. Furthermore we will remove all special characters, emojis and stopwords. 

# Methods Used