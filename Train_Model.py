# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:14:49 2020

@author: SParravano
"""
dir_path='/dsdata/Soft_Cosine'
#%%
import os
import pandas as pd
import nltk
import gensim


import urllib.request
from urllib.request import urlretrieve
from bs4 import BeautifulSoup as soup

import time
import re
import string

from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
main_pages=os.path.join(dir_path,'Text_Data','Goal_website','Main_Pages')
text_to_train=os.path.join(dir_path,'Text_Data','Goal_website','Output_Text','Body_of_Text.txt')
#%%

while True:
    instruction=str(input('Do you want to read file line by line?  (Y/N): ')).lower()
    if not (instruction== 'y' or instruction=='n'):
        print('Sorry, You did not enter a valid response. Please enter Y or N :)')
        continue
    else:
        print('Thank you for the input. It will help correctly define the course of action..')
        break
#%%
#Gensim only requires that the input must provide sentences sequentially, when iterated over.
#No need to keep everything in RAM: we can provide one sentence, process it, forget it, load another sentence…
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                yield line.split()
 

#%%

if instruction =='N':
    sentences = MySentences('/some/directory') # a memory-friendly iterator
    model = gensim.models.Word2Vec(sentences)
else:
    with open(text_to_train) as f:
        sentences=f.readlines()
    sentences_list=[re.sub(r'\n','',x) for x in sentences]
    sentences_list=[re.sub(r'\'\'','',x) for x in sentences_list]
    #sentences_list=[[x] for x in sentences_list] 
    sentences_list=[x.split(' ') for x in sentences_list]    
    start_time=time.time()
    print(sentences_list[0:5])
    print('Training Model...')
    model = gensim.models.Word2Vec(sentences_list, min_count=5,window=5,workers=80)
    endtime=time.time()
    training_time=endtime-start_time
    print('model was trained in:',str(endtime-start_time),'seconds')
    model.save(os.path.join(dir_path,'Text_Data','word2VecModel','Goal_Model'))
    model['messi']
