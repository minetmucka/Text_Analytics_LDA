#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import required packages
import pickle # for saving and loading objects
import gensim
from gensim.models import CoherenceModel
import gensim.corpora as corpora
from nltk.tag import pos_tag
from gensim.utils import simple_preprocess
import numpy as np
import pandas as pd
import os
import time
import datetime
import tarfile
import re
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from collections import OrderedDict

#Extracting an open source dataset from the New York Times
tar = tarfile.open("20news-bydate.tar.gz")
tar.extractall()
tar.close()
##Make a list of the folders in the dataset
directory = [f for f in os.listdir('./20news-bydate-train') if not f.startswith('.')]
dt_stamp = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H_%M_%S')

texts=[]
text_corpus=[]
texts_mod=[]
#Import Newsgroups Text Data
for i in range(len(directory)):
    #if(directory[i]==('rec.sport.baseball') or directory[i]==('rec.sport.hockey')):
    ##Create a list of files in the given dictionary 
        files = os.listdir('./20news-bydate-train/' + directory[i])

        for j in range(len(files)):     

            ##Path of each file 
            path = './20news-bydate-train/' + directory[i] + '/' + files[j]

            ##open the file and read it
            text1 = open(path, 'r', errors='ignore').read()        
            texts.append(text1)

# Data Cleansing
for text1 in texts:
    text = text1
    #Remove emails and newline characters
    # Remove Emails
    text = re.sub('\S*@\S*\s?', '', text)

    # Remove new line characters
    text = re.sub('\s+', ' ', text) 

    # Remove distracting single quotes
    text = re.sub("\'", "", text) 
    
    #remove_names()
    new_sentence = []
    text_split=text.split(" ")
    tagged_sentence = pos_tag([word for word in text_split if word])
    for word, tag in tagged_sentence:
        if tag in ['NNP', 'NNPS']:
            lemma_word = ""
        else:
            lemma_word = word

        new_sentence.append(lemma_word)
    text2=""
    for i in new_sentence:
        text2 = text2 + " " + i
    text=text2

    # Converting to Lower case
    text=text.lower()

    # Removing unwanted information 
    text=text.replace("-"," ").replace(".com"," ")
    text=' '.join(re.sub("(@[A-Za-z[0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\/S+)", " ", text).split())

    # Adding Space at begining and End of the Text
    text=" " + text + " "

    #replace_incomplete_word()
    #Lemmatization
    #smart_lemmatize()
    new_sentence = []
    lemma = WordNetLemmatizer()
    text_split=text.split(" ")
    tagged_sentence = pos_tag([word for word in text_split if word])
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
    text2=""
    for i in new_sentence:
        text2 = text2 + " " + i
    text=text2

    #removeUnicode()
    text = re.sub(r'(\\u[0-9A-Fa-f]+)',r' ', text)       
    text = re.sub(r'[^\x00-\x7f]',r' ',text)

    #replaceURL()
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)
    text = re.sub(r'#([^\s]+)', r'\1', text)

    #replaceAtUser()
    text = re.sub('@[^\s]+',' ',text)

    #removeHashtagInFrontOfWord()
    text = re.sub(r'#([^\s]+)', r'\1', text)

    #removeNumbers()
    text = ''.join([i for i in text if not i.isdigit()]) 

    #replaceMultiExclamationMark()
    text = re.sub(r"(\!)\1+", ' ', text)

    #replaceMultiQuestionMark()
    text = re.sub(r"(\?)\1+", ' ', text)

    #replaceMultiStopMark()
    text = re.sub(r"(\.)\1+", '.', text)

    #replace_incomplete_word()
    #remove_stop_words()
    from nltk.corpus import stopwords
    rmv_wrd_lst=stopwords.words('english')
    rmv_stop_word=[]
    for i in rmv_wrd_lst:
        #print(i)
        wrd=" " + str(i) + " "
        rmv_stop_word.append(wrd)
    for t in rmv_stop_word:
        text=text.replace(t," ")

    #replaceMultiSpace()
    #text=text.replace("  "," ")
    text = re.sub(r"(\ )\1+", ' ', text)        
    #Clean-up using gensim’s simple_preprocess()
    #text_corpus.append(gensim.utils.simple_preprocess(str((text.split()))))
    text_corpus.append(gensim.utils.simple_preprocess(str(text.split()),deacc=True))
    texts_mod.append(text)
    
print(len(texts_mod))

vectorizer = TfidfVectorizer(analyzer='word',stop_words=rmv_stop_word)
data_vectorized = vectorizer.fit_transform([' '.join(x) for x in text_corpus])
search_params = {'n_components': [5,10,15,20], 'learning_decay': [.5, .7, .9]}
lda = LatentDirichletAllocation()
model = GridSearchCV(lda, param_grid=search_params)
model.fit(data_vectorized)
best_lda_model = model.best_estimator_

print("best model params", model.best_params_)
# Log Likelihood Score
print("Best Log Likelihood Score: ", model.best_score_)
# Perplexity
print("Model Perplexity: ", best_lda_model.perplexity(data_vectorized))

#Alternative approach Coherence Score
limit=21
start=2
step=1
coherence_values = []
model_list = []
i_model_list = []
i_cv_list = []
dictionary=corpora.Dictionary(text_corpus)
corpus=[dictionary.doc2bow(text) for text in text_corpus]
#Coherence_values vs no_topics
for num_topics in range(start, limit, step):
            print('Number of Topics', num_topics)
            model = gensim.models.LdaModel(corpus,num_topics=num_topics, id2word=dictionary,random_state=100,chunksize=10000,passes=2,alpha='auto')
            
            coherencemodel = CoherenceModel(model=model, texts=text_corpus,corpus=corpus,dictionary=dictionary, coherence='c_v')
            print(' CV ' , coherencemodel.get_coherence())
            coherence_values.append(coherencemodel.get_coherence())
            model_list.append(num_topics)
            
#Coherence Plot                 
get_ipython().run_line_magic('matplotlib', 'inline')
x=range(start,limit,step)
plt.plot(x,coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence Score")
plt.legend("Coherence_values",loc='best')
plt.grid()
work_dir=os.getcwd()
title_name="CountVectorizer_20_newsgroup_plot"
img_name=str(work_dir)+'/'+str(title_name)+ str(".png")
#fig=plt.figure(figsize=(6,4))
plt.savefig(img_name,dpi=300,bbox_inches='tight')
plt.show()
plt.clf()

# Saving the Final LDA Model and Data as Pickle file
with open ('text_corpus_20Newgroup_' + dt_stamp + '.pkl', 'wb') as pkl_fl:
        pickle.dump(text_corpus,pkl_fl)
        
with open ('lda_model_20Newgroup_' + dt_stamp + '.pkl', 'wb') as pkl_fl:
        pickle.dump(lda_model,pkl_fl)
        
with open ('texts_20Newgroup_' + dt_stamp + '.pkl', 'wb') as pkl_fl:
        pickle.dump(texts,pkl_fl)     
        
with open ('text_mod_20Newgroup_' + dt_stamp + '.pkl', 'wb') as pkl_fl:
        pickle.dump(texts_mod,pkl_fl)

with open ('data_vectorized_20Newgroup_' + dt_stamp + '.pkl', 'wb') as pkl_fl:
        pickle.dump(data_vectorized,pkl_fl)
        
dictionary=corpora.Dictionary(text_corpus)
corpus=[dictionary.doc2bow(text) for text in text_corpus]
lda_model =  gensim.models.ldamodel.LdaModel(corpus=corpus,id2word=dictionary,num_topics=10,update_every=1,chunksize=10000,passes=2,random_state=100)

pyLDAvis.enable_notebook()
viz=pyLDAvis.gensim.prepare(lda_model,corpus,dictionary,sort_topics=False)
pyLDAvis.display(viz)

# save the LDA visualization
dt_stamp = datetime.datetime.strftime(datetime.datetime.now(),'%Y%m%d_%H_%M_%S')
pyLDAvis.save_html(viz,'lda_visualization_20Newgroup_10_'+ dt_stamp + '.html')

