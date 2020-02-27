#!/usr/bin/env python
# coding: utf-8

# ## Packages you may need to install prior to run
# 
# 1. Gensim (https://pypi.org/project/gensim/ | https://radimrehurek.com/gensim/) -- LDA package
# 2. NLTK (https://pypi.org/project/nltk/ | https://www.nltk.org/) -- used for pre-processing 
# 3. pyLDAvis (https://pypi.org/project/pyLDAvis/ | https://pyldavis.readthedocs.io/en/latest/) -- used for visualing LDA analysis
# 
# Data Source:  
# https://www.kaggle.com/kulgen/elon-musks-tweets

# ## Import packages

# In[1]:


import pickle # for saving and loading objects
import time
import datetime
from collections import OrderedDict, Counter
import pandas as pd
import numpy as np
import string
import re
import itertools

# gensim methods
import gensim 
from gensim.models import CoherenceModel # used to calculate Coherence score
import gensim.corpora as corpora

# NLTK methods
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import words
from nltk.tag import pos_tag

# pyLDAvis methods
import pyLDAvis
import pyLDAvis.gensim 

# graphing tools
import seaborn as sns

# turns warnings off 
import warnings
warnings.filterwarnings("ignore")


# ## Read in Data

# In[2]:


df = pd.read_excel('C:\\Users\\muckam\\Desktop\\elonmusk_topicmodeling\\data_elonmusk.xlsx')
df.head()


# In[3]:


text_series = df['Tweet'] # create a series that contains only documents
print(text_series.iloc[4]) 


# ## Clean up data... 

# In[4]:


# translate every punctuation to space
punct_dict = dict.fromkeys(string.punctuation,' ')
punct_translator = str.maketrans(punct_dict) 

# build a set of stopwords
stopword_set = set(stopwords.words('english')) # uses NLTK stopwords 

# define a stemmer 
stemmer = SnowballStemmer('english')

# define a lemmatization function 
lemma = WordNetLemmatizer()

def smart_lemmatize(sentence, lemma):
    """Detects the part of speech for each word and then lemmatizes using that tag if available (default to Noun)

   Returns
   -------
   list
       list of lemmatized words

   Examples
   --------
   >>> self.smart_lemmatize(['testing','this','out'])
   ['test','this','out']
    """
    new_sentence = []
    tagged_sentence = pos_tag([word for word in sentence if word])
    for word, tag in tagged_sentence:
        if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            pos = 'n'
        elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            pos = 'v'
        elif tag in ['JJ', 'JJR', 'JJS']:
            pos = 'a'
        elif tag in ['RB', 'RBR', 'RBS']:
            pos = 'r'
        else:
            pos = 'n'
        lemma_word = lemma.lemmatize(word=word, pos=pos)
        new_sentence.append(lemma_word)
    return new_sentence

def name_detector(sentence):
    pass
    


# In[5]:


# CHANGE OUT SENTENCE WITH DOCUMENT AND CORPUS ETC.

# loop through each document using list comprehensions
documents = text_series.copy()
# remove hyperlinks
documents = [re.sub(r'http\S+', '', sentence) for sentence in documents]
# remove emails
documents = [re.sub('\S*@\S*\s?', '', sentence) for sentence in documents]
# remove newline chars
documents = [re.sub('\s+', ' ', sentence) for sentence in documents]
# remove single quotes
documents = [re.sub("\'", "", sentence) for sentence in documents]
# remove numbers
documents = [re.sub("\d+", "", sentence) for sentence in documents]
# remove punctuation
documents = [sentence.lower().translate(punct_translator).split() for sentence in documents]
# remove stop words
documents = [[word for word in sentence if word not in stopword_set] for sentence in documents]
# lemmatize 
documents = [smart_lemmatize(sentence, lemma) for sentence in documents]
# stemming 
# documents = [[stemmer.stem(word) for word in sentence] for sentence in documents]

# remove names
#documents = [re.sub('\s+', ' ', sentence) for sentence in documents]

# remove any new stopwords created by lemmatization / stemming
documents = [[word for word in sentence if word not in stopword_set] for sentence in documents]

# remove blank strings
documents = [sentence for sentence in [[word for word in sentence if word] for sentence in documents]]


# In[6]:


print(documents[4])


# In[7]:


# merge back with dataframe 
df['CLEAN_DOCUMENTS'] = documents

# calculate length of columns
df['CLEAN_LENGTH'] = df['CLEAN_DOCUMENTS'].apply(lambda x: len(x))
df.loc[df['CLEAN_LENGTH'] <= 5, 'CLEAN_LENGTH'].value_counts(sort=False)


# In[8]:


# only want columns with data
df = df.loc[df['CLEAN_LENGTH'] > 3]

# preview
df.head()


# ## Create bigrams (optional)

# In[9]:


bigram_config = gensim.models.Phrases(df['CLEAN_DOCUMENTS'].tolist(), min_count=5, threshold=1)
bigram_model = gensim.models.phrases.Phraser(bigram_config)

bigram_documents = [bigram_model[document] for document in df['CLEAN_DOCUMENTS'].tolist()]
bigram_documents = [[lemma.lemmatize(word) for word in sentence] for sentence in bigram_documents]

df['BIGRAM_DOCUMENTS'] = bigram_documents


# ## Convert each word to an integer, create dictionary with integer word pairings

# In[11]:


gs_dictionary = gensim.corpora.Dictionary(df['BIGRAM_DOCUMENTS'].tolist()) 

print('key 1',gs_dictionary[1])
print('key 100',gs_dictionary[100])


# ## Create a bag of words, convert list of words to tuples with integer key and frequency of occurence in document

# In[12]:


bow_corpus = [gs_dictionary.doc2bow(doc) for doc in df['BIGRAM_DOCUMENTS'].tolist()]
bow_corpus[4]


# ## 

# In[13]:


# create a function to calculate coherence value for an LDA model

'''
coherence definition: 

'''

def calculate_coherence_value(lda_model):
        """
        Calculates coherence value for a given model

        Parameters:
        ----------
        lda_model (object): an LDA model to calculate coherence value for

        Returns:
        -------
        coherence value (float): a coherence value, greater is better
        """

        coherence_model = CoherenceModel(model=lda_model,
                                        texts = df['BIGRAM_DOCUMENTS'].tolist(),
                                        dictionary = gs_dictionary,
                                        coherence = 'c_v')

        return coherence_model.get_coherence()


# ## Build an LDA model 
# ### Step 1: Figure out your topic count

# In[14]:


coherence_value_dict = {}
for i in range(2,30,2):
    print('Calculating Coherence Value for topic count = {}'.format(str(i)))
    # step 1 -- build LDA model
    lda_model = gensim.models.LdaModel(corpus=bow_corpus,
                                       id2word = gs_dictionary,
                                       num_topics = i,
                                       random_state = 42,
                                       alpha = 'auto', eta = 'auto')

    # step 2 -- calculate coherence value
    coherence_value = calculate_coherence_value(lda_model)
    print('\tCoherence Value: {}'.format(str(round(coherence_value,2))))

    # step 3 -- update dictionary
    coherence_value_dict[i] = {'Coherence Value':coherence_value}

    # store coherence values in df
    topic_count_coherence_df = pd.DataFrame(coherence_value_dict).T


topic_count_coherence_df.sort_values(by = 'Coherence Value',ascending=False)


# In[15]:


ax = sns.lineplot(x="index", y="Coherence Value", data=topic_count_coherence_df.reset_index())


# ### Step 2: Brute Force Hypertune Parameters (optional)
# 
# Alpha definition: 
# 
# Beta definition: 

# In[21]:


hypertuning_dict = OrderedDict()

alpha_list = ['auto',0.01]
eta_list = ['auto',0.25, 0.5, 1, 5]
num_topics = 18

# get all possible eta / alpha combinations
alpha_eta_combo = list(itertools.product(*[alpha_list,eta_list]))

for idx, alpha_eta_tup in enumerate(alpha_eta_combo): # loop through combinations
    alpha, eta = alpha_eta_tup

    print('Evaluating alpha = {}, eta = {}'.format(str(alpha),str(eta)))
    hyper_lda_model = gensim.models.LdaModel(corpus=bow_corpus,
                                       id2word = gs_dictionary,
                                       num_topics = num_topics,
                                       random_state = 1,
                                       alpha = alpha,
                                       eta = eta,
                                       
                                       )

    coherence_value = calculate_coherence_value(hyper_lda_model)
    print('\tCoherence Value: {}'.format(str(round(coherence_value,2))))
    # store values in dictionary
    hypertuning_dict[idx] = {'number of topics':num_topics,
                            'alpha':alpha,'eta':eta,
                            'coherence value':coherence_value}

hypertuned_df = pd.DataFrame(hypertuning_dict).T # convert to DataFrame
hypertuned_df.sort_values(by='coherence value',ascending=False)


# ### Step 3: Build model 

# In[20]:


lda_model = gensim.models.LdaModel(corpus=bow_corpus,
                                   id2word = gs_dictionary,
                                   num_topics = 18, # get from elbow chart above 
                                   random_state = 42,
                                   )


# ### Step 4: Visualize model 
# 
# works best in chrome

# In[22]:


pyLDAvis.enable_notebook() # code for Jupyter notebooks
lda_visualization = pyLDAvis.gensim.prepare(lda_model, bow_corpus, gs_dictionary, sort_topics = False)


# In[23]:


lda_visualization

