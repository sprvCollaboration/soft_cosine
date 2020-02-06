# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 11:28:09 2020

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
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

import multiprocessing as mp
import numpy as np

main_pages=os.path.join(dir_path,'Text_Data','Goal_website','Main_Pages')

#%%
#let's first download the main pages. We will then use these to parse out the html of the specific articles.
#I will then download the individual articles

#%%
#let's download a bunch of articles from Goal.com


def Main_Page_Download_FN(base_url,lower_bound_page,upper_bound_page,output_dir,error_dir):
    base_url = base_url
    page_ending=[str(x) for x in range(lower_bound_page,upper_bound_page)]
    startTime = time.time()
    for i in page_ending:
        try:
            print('Downloading page:',base_url+i)
            page = urllib.request.urlopen(base_url+i)
            page_raw_html=page.read()
            page.close()
            file_name=base_url+'_'+str(i)
            file_name=re.sub('https://','',file_name)
            file_name=re.sub('\.','_',file_name)
            file_name=re.sub('\?','_',file_name)
            file_name=re.sub('\/','-',file_name)
            file_name=re.sub('=','',file_name)
            file_name=file_name+'.txt'
            f=open(os.path.join(output_dir,file_name),'w+')
            f.write(str(page_raw_html))
            print('Page saved to directory. Parsing initiated to extract url....')
            #need to save page as a soup object
            page_soup=soup(page_raw_html,"html.parser")
            #we now parse and keep class with url information for each product link
            url_block=page_soup.select('a')
            #this is by default a list..we iterate through the list to extract the individual urls
            #url_list=[url_block[x].select('a')[0]['href'] for x in range(0,len(url_block))]
            url_list=[url_block[x]['href'] for x in range(0,len(url_block))]
            url_list_final=[]
            for x in url_list:
                if x.startswith('/en-us') and not (x.endswith('/en-us') or x.endswith('live-scores') or
                                x.endswith('privacy-policy') or x.endswith('terms-conditions')
                                or x.endswith('contact') or x.endswith('2')):
                    url_list_final.append(x)
            #we need to augment the urls by adding the proper prefix
            proper_prefix='https://www.goal.com'
            url_list_final=[proper_prefix + x for x in url_list_final]
            #we write the urls to a csv..we will then read these in 1 by one to execute the workflow we need for each of the products
            url_file=os.path.join(dir_path,'Text_Data','Goal_website','URL_LIST.csv')
            if os.path.exists(url_file)==True:
                url_file = open(os.path.join(dir_path,'Text_Data','Goal_website','URL_LIST.csv'),'a')
                for r in url_list_final:
                    url_file.write(r +','+(base_url+i)+"\n")
                url_file.close()
            else:
                url_file = open(os.path.join(dir_path,'Text_Data','Goal_website','URL_LIST.csv'),'w')
                for r in url_list_final:
                    url_file.write(r +','+(base_url+i)+ "\n")
                url_file.close()
            print('URL extracted and stored in proper format.')
            print('\n')
        except:
            print('Error downloading Main Page: ',base_url+i)
            error_log_main=os.path.join(error_dir,'Error_log_Main_Page_Download_Fail.csv')
            if os.path.exists(error_log_main)==True:
                error_log_main = open(os.path.join(error_dir,'Error_log_Main_Page_Download_Fail.csv'),'a')
                error_log_main.write(base_url+i + "\n")
                error_log_main.close()
            else:
                error_log_main = open(os.path.join(error_dir,'Error_log_Main_Page_Download_Fail.csv'),'w')
                error_log_main.write(base_url+i + "\n")
                error_log_main.close()
    endTime=time.time()
    print('The main pages with links to scrape were downloaded in: ',endTime-startTime,'seconds')

#--- function to dpwnload page that we will scrape
def Pages_to_scrape_download_FN(url_to_download,output_dir,error_dir):
    try:
        print('Downloading Page: ',url_to_download)
        page_d=urllib.request.urlopen(url_to_download)
        page_d_raw_html=page_d.read()
        page_d.close()
        file_name_page=re.sub('https://','',url_to_download)
        file_name_page=re.sub('/','-',file_name_page)
        file_name_page=re.sub('\.','_',file_name_page)
        file_name_page=file_name_page+'.txt'
        f=open(os.path.join(output_dir,file_name_page),'w+')
        f.write(str(page_d_raw_html))
        print('Page saved to directory: ',os.path.join(output_dir,file_name_page))
    except:
        print('Error with this page: '+url_to_download+'.Did not download.We will write this to our error log. Moving on to next Page....')
        error_log_file=os.path.join(error_dir,'Error_Log_File_URL_Download_Fail.csv')
        if os.path.exists(error_log_file)==True:
            error_log_file = open(os.path.join(error_dir,'Error_Log_File_URL_Download_Fail.csv'),'a')
            error_log_file.write(url_to_download + "\n")
            error_log_file.close()
        else:
            error_log_file = open(os.path.join(error_dir,'Error_Log_File_URL_Download_Fail.csv'),'w')
            error_log_file.write(url_to_download + "\n")
            error_log_file.close()



#%%

#Parallel implementation
################################################################
startTime = time.time()

cpus_to_use=mp.cpu_count()
print('We will use: '+str(cpus_to_use)+' processors')

num_of_processors=cpus_to_use
# we want to set up processes to feed to our paralellization workflow.
# we need to organize our inputs in the meta_files_list into a list of lists
list_of_pages=[x for x in range(1,200000)]
processes_list=np.array_split(np.array(list_of_pages),num_of_processors)
processes_list=[list(x) for x in processes_list]
print('Downloading URL list and saving to file')
print('######## EXECUTING PARALLEL WORKFLOW #####################')
#as an input this takes an element of process_list (which in turn is a list of files to proceess)
def run_process(process):
    for i in range(len(process)):
        Main_Page_Download_FN(base_url='https://www.goal.com/en-us/news/',
                      lower_bound_page=i,
                      upper_bound_page=i+1,
                      output_dir=os.path.join(dir_path,'Text_Data','Goal_website','Main_Pages'),
                      error_dir=os.path.join(dir_path,'Text_Data','Goal_website','Error_Log'))


pool=mp.Pool(processes=len(processes_list))
pool.map(run_process,processes_list)

endTime=time.time()
print('')
##################################################################


#print('Downloading URL list and saving to file')
#Main_Page_Download_FN(base_url='https://www.goal.com/en-us/news/',
#                      lower_bound_page=1,
#                      upper_bound_page=200000,
#                      output_dir=os.path.join(dir_path,'Text_Data','Goal_website','Main_Pages'),
#                      error_dir=os.path.join(dir_path,'Text_Data','Goal_website','Error_Log'))

################################################################

print('With the links extracted we can now download each page containing articles...')

list_of_urls=list(pd.read_csv(os.path.join(dir_path,'Text_Data','Goal_website','URL_LIST.csv'),header=None)[0])
#becuase the urls are often duplicated we will take unique urls only
list_of_urls=list(set(list_of_urls))

print('We need to download and scrape: ',len(list_of_urls), 'Web pages')

cpus_to_use=mp.cpu_count()-1
print('We will use: '+str(cpus_to_use)+' processors')

#Parallel implementation
################################################################
startTime = time.time()

cpus_to_use=mp.cpu_count()
print('We will use: '+str(cpus_to_use)+' processors')

num_of_processors=cpus_to_use
# we want to set up processes to feed to our paralellization workflow. 
# we need to organize our inputs in the meta_files_list into a list of lists
processes_list=np.array_split(np.array(list_of_urls),num_of_processors)
processes_list=[list(x) for x in processes_list]

print('######## EXECUTING PARALLEL WORKFLOW #####################')
#as an input this takes an element of process_list (which in turn is a list of files to proceess)
def run_process(process):
    for i in range(len(process)):
        Pages_to_scrape_download_FN(process[i],
                                output_dir=os.path.join(dir_path,'Text_Data','Goal_website','Pages_to_Scrape'),
                                error_dir=os.path.join(dir_path,'Text_Data','Goal_website','Error_Log'))


pool=mp.Pool(processes=len(processes_list))
pool.map(run_process,processes_list)

endTime=time.time()
print('')
##################################################################

#startTime = time.time()
#for i in range(0,len(list_of_urls)): #need to change range here to len(list_of_urls)
#    Pages_to_scrape_download_FN(list_of_urls[i],
#                                output_dir=os.path.join(dir_path,'Text_Data','Goal_website','Pages_to_Scrape'),
#                                error_dir=os.path.join(dir_path,'Text_Data','Goal_website','Error_Log'))

#endTime=time.time()
print('')
print('The pages we need to scrape were downloaded in: ',endTime-startTime,'seconds') 
#%%


def text_processing(text_raw):
    text_proc=re.sub('\n','',text_raw)
    text_proc=re.sub('\t','',text_proc)
    #make all lower
    text_proc=text_proc.lower()
    text_proc=re.sub(r'\(',' ',text_proc)
    text_proc=re.sub(r'\)',' ',text_proc)
    text_proc=re.sub(r'\/',' ',text_proc)
    text_proc=re.sub(r'\\',' ',text_proc)
    text_proc=re.sub(r'\'',' ',text_proc)
    #remove leading and trailing white space
    text_proc=text_proc.strip()  
    word_tokens=word_tokenize(text_proc)
    list_of_bad_tokens=['\\xe2\\x80\\x99s','\\xe2\\x80\\x9d','\\xe2\\x80\\x9cI','\\xe2\\x80\\x99','\\xe2\\x80\\x9c',
                        '\\n\\xe2\\x80\\x9c','\\n\\xe2\\x80\\x9c','\\xe2\\x80\\x98','xe2','x80','x9d','x9ci','x99ll',
                        'xc2','xao']
    tokens=[t for t in word_tokens if not any( x in t for x in list_of_bad_tokens)]
    #stopWords = stopwords.words('english')    
    #uncomment above and comment below if we don't want stop words
    stopWords=[]    
    keywords_keep=[word for word in tokens if word not in stopWords]
    keywords_keep=[re.sub('\\\\n','',x) for x in keywords_keep]
    text_proc=' '.join(keywords_keep)
    printable_char=set(string.printable)
    text_proc=''.join(filter(lambda x: x in printable_char, text_proc))
    text_proc=re.sub('\\\\n','',text_proc)
    text_proc=re.sub('\\\\',' ',text_proc)
    #remove redundant white space
    text_proc=text_proc.strip()
    text_proc=re.sub(' +',' ',text_proc)
    tokens_sent=sent_tokenize(text_proc)
    #we now remove punctuation
    #stem words - porter stemming
    #stemmer=PorterStemmer()
    #tokens=word_tokenize(text_proc)
    #text_proc=' '.join([stemmer.stem(x) for x in tokens])
    return(tokens_sent)
#%%
    
#with the pages downloaded we can now focus on parsing each web page and extracting the text


def Goal_scraper_FN(page_to_scrape,output_file,error_dir):
    try:
        print('Parsing page:',str(page_to_scrape))
        startTime = time.time()
        with open(page_to_scrape) as html_file:
            page_0_soup=soup(html_file,'lxml')
        #print(page_0_soup.prettify())
        #page_scraped=page_to_scrape
        
        body_text=page_0_soup.find('div',class_='body').text
        body_text=body_text
        #write code to preprocess the text data. Clean non printable characters
        tokens_sent=text_processing(body_text)
        # write each sentence to a text file. One sentence per line
        with open(output_file,'a+') as output:
            for line in tokens_sent:
                output.write("{}\n".format(line))
        print('scraped and processed text from:',str(page_to_scrape))
        print('SUCCESSSSSS')
        print('##########')
        endTime=time.time()
        print('The page was scraped/parsed in: ',endTime-startTime,'seconds')
    except Exception as e:
        print('Error with this page: '+page_to_scrape+'.Did not Scrape.We will write this to our error log. Moving on to next Page....')
        print('ERROR:',str(e))
        error_log_file=os.path.join(error_dir,'Error_Log_Page_Parsing_Fail.csv')
        if os.path.exists(error_log_file)==True:
            error_log_file = open(os.path.join(error_dir,'Error_Log_Page_Parsing_Fail.csv'),'a')
            error_log_file.write(page_to_scrape + "\n")
            error_log_file.close()
        else:
            error_log_file = open(os.path.join(error_dir,'Error_Log_Page_Parsing_Fail.csv'),'w')
            error_log_file.write(page_to_scrape + "\n")
            error_log_file.close()

#%%
pages_to_scrape_list=os.listdir(os.path.join(dir_path,'Text_Data','Goal_website','Pages_to_Scrape'))


##################################################
startTime = time.time()
processes_list=np.array_split(np.array(pages_to_scrape_list),num_of_processors)
processes_list=[list(x) for x in processes_list]

print('######## EXECUTING PARALLEL WORKFLOW #####################')
#as an input this takes an element of process_list (which in turn is a list of files to proceess)
def run_process(process):
    for i in range(len(process)):
        Goal_scraper_FN(os.path.join(dir_path,'Text_Data','Goal_website','Pages_to_Scrape',process[i]),
                    output_file=os.path.join(dir_path,'Text_Data','Goal_website','Output_Text','Body_of_Text.txt'),
                    error_dir=os.path.join(dir_path,'Text_Data','Goal_website','Error_Log'))

pool=mp.Pool(processes=len(processes_list))
pool.map(run_process,processes_list)

endTime=time.time()
print('')


#startTime = time.time()
#for i in range(0,len(pages_to_scrape_list)): 
#    Goal_scraper_FN(os.path.join(dir_path,'Text_Data','Goal_website','Pages_to_Scrape',pages_to_scrape_list[i]),
#                    output_file=os.path.join(dir_path,'Text_Data','Goal_website','Output_Text','Body_of_Text.txt'),
#                    error_dir=os.path.join(dir_path,'Text_Data','Goal_website','Error_Log'))
#
#endTime=time.time()
#print('')
print('The pages we need to scrape were downloaded in: ',endTime-startTime,'seconds') 
#%%
