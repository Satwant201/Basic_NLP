import pandas as pd
import re
import numpy as np
from collections import Counter
import matplotlib
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk.stem import PorterStemmer
import pandas as pd
import re
import nltk
from gensim import models, corpora
from nltk import word_tokenize
from nltk.corpus import stopwords


########## Data cleaning for Topic modelling 
def data_cleaning(tweet,custom_list):
    tweet = re.sub(r'\$\w*','',tweet) # Remove tickers
    tweet = re.sub(r'https?:\/\/.*\/\w*','',tweet) # Remove hyperlinks
    tweet = re.sub(r'['+string.punctuation+']+', ' ',tweet) # Remove puncutations like 's
    #stop = set(stopwords.words('english'))
    tweet = re.sub(r'[^a-zA-Z0-9]'," ",tweet)
    stop_words=set(['a',    'about', 'above', 'after',   'again',  'against',              'ain',      'all',        'am',               'an',       'and',     'any',     'are',      'as',        'at',        'be',       'because',            'been',   'before',               'being',  'below', 'between',           'both',   'but',      'by',        'can',     'couldn',               'd',               'did',      'didn',    'do',       'does',   'doesn', 'doing',  'don',     'down',  'during',               'each',               'few',     'for',      'from',   'further',              'had',     'hadn',   'has',      'hasn',   'have',   'haven',               'having',               'he',       'her',      'here',    'hers',    'herself',              'him',     'himself',               'his',       'how',    'i',           'if',         'in',         'into',     'is',         'isn',       'it',         'its',        'itself',               'just',     'll',          'm',        'ma',      'me',      'mightn',              'more',  'most',   'mustn', 'my',               'myself',               'needn', 'now',    'o',         'of',        'off',      'on',       'once',   'only',    'or',               'other',  'our',      'ours',    'ourselves',          'out',      'over',    'own',    're',        's',          'same',               'shan',   'she',      'should',               'so',        'some',  'such',    't',          'than',    'that',    'the',               'their',   'theirs',  'them',  'themselves',      'then',    'there',  'these',  'they',    'this',     'those',               'through',            'to',        'too',     'under', 'until',    'up',       've',        'very',    'was',     'we',               'were',   'weren',               'what',   'when',  'where',               'which', 'while',  'who',    'whom',               'why',    'will',      'with',    'won',    'y',          'you',     'your',    'yours',  'yourself',               'yourselves'])
    exclude = set(string.punctuation)
    exclude1= set(custom_list)
    stop_words.update(exclude1)
    lemma = WordNetLemmatizer()
    stop_free = " ".join([i for i in tweet.lower().split() if i not in stop_words])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized


############ Data cleaning for Ngram analysis 
def clean_text(text,custom_list):
    tokenized_text = word_tokenize(text.lower())
    cleaned_text = [t for t in tokenized_text if t not in STOPWORDS and re.match('[a-zA-Z\-][a-zA-Z\-]{2,}', t)]
    cleaned_text1 = [t for t in cleaned_text if t not in custom_list]
    return cleaned_text
	
########## Pass custom list of stop words in the following list "custom_list"
	
custom_list=["fall"]

############## Creating UDF for ngram analysis , n means number of grams 1,2,3 ....
def cal_ngram(text,n):
    token = nltk.word_tokenize(text)
    n_grams = ngrams(token,n)
    return n_grams


####### Load input data for text analysis 
text1 = pd.read_csv(r'path.csv', sep=',', encoding='iso-8859-1')

text1.dropna(axis=0,how='any',inplace=True)

#### Here "Text" column is our target column 
text=text1["Text"].values.tolist()



a=0

####### mention number of topics 
NUM_TOPICS = 10
STOPWORDS = stopwords.words('english')

### text column was converted into a list for easy iterations 
tweet_text=text  # text is a list of column to be modelled
doc_complete=tweet_text
#doc_complete_2=[]
doc_complete_2=tweet_text
 
 
# For gensim we need to tokenize the data and filter out stopwords
tokenized_data = []
for w in doc_complete_2:
    a=a+1
    tokenized_data.append(data_cleaning(w,custom_list))
# Build a Dictionary - association word to numeric id
dictionary = corpora.Dictionary(tokenized_data)
# Transform the collection of texts to a numerical form
corpus = [dictionary.doc2bow(text) for text in tokenized_data] 
# Have a look at how the Nth document looks like: [(word_id, count), ...]
# [(12, 3), (14, 1), (21, 1), (25, 5), (30, 2), (31, 5), (33, 1), (42, 1), (43, 2),  ...
 # Build the LDA model
lda_model = models.LdaModel(corpus=corpus, num_topics=NUM_TOPICS, id2word=dictionary)
final=[]
for topic in lda_model.show_topics(num_topics=NUM_TOPICS, formatted=False, num_words=6):
    topicwords = [w for (w, val) in topic[1]]
    topicwords_val = [val for (w, val) in topic[1]]
    final.append([topicwords,topicwords_val])
final1=pd.DataFrame(final,columns=["topic","prob"])



final1.to_csv(path+"/topics.csv")




########################################N-gram analysis

import re
import string
from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from nltk import ngrams






n_gram = cal_ngram(text[0],2)   # generating N grams for input data

n_gram_common = Counter(n_gram).most_common()  # Listing top 10 trending N grams
n_gram_df=pd.DataFrame(n_gram_common,columns=["N Gram","Frquency"])
