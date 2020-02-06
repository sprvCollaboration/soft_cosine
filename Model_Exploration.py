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


model=gensim.models.Word2Vec.load(os.path.join(dir_path,'Text_Data','word2VecModel','Goal_Model'))

print(model.most_similar('messi',topn=10))
