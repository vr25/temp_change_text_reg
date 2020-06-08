from nltk.tokenize import word_tokenize
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr
import pandas as pd
import csv
import time
import gensim
import gensim.models as g
import logging
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import os
import sys

start = time.time()

def cos_similarity(a,b):
    n = len(a)
    #print("cosine sim n = ", n)
    a = np.asarray(a).reshape(1,n)
    b = np.asarray(b).reshape(1,n)
    cos_lib = cosine_similarity(a, b)
    return(cos_lib[0][0])


#create train_docs corpus
df = pd.read_csv("text_similarity_scores.csv")

'''
#uncomment this!
f = open('train_docs_sec7.txt', 'w+') 

cik_year1_list = df['cik_year1'].tolist()  
cik_year2_list = df['cik_year2'].tolist() 

print("lengths: ", len(cik_year1_list), len(cik_year2_list))

basepath1 = "/data/ftm/xgb_regr/ch_an_data/cleaned_sec7/"
files_list = os.listdir(basepath1)
#print("len files list: ", len(set(files_list)), files_list[:2])
files_list_cik_year = [i.split("_")[1] + "_" + i.split("_")[0].split("-")[0] for i in files_list]

files_loc_dict = dict(zip(files_list_cik_year, files_list))

print("length: ", len(files_loc_dict))

c = 0

for i in range(len(cik_year1_list)):

    try:
        print("cik_1: ", files_loc_dict[cik_year1_list[i]])
        print("cik_2: ", files_loc_dict[cik_year2_list[i]])
        print("-------------------------------------")

        i_con = open(basepath1 + files_loc_dict[cik_year1_list[i]]).read()
        j_con = open(basepath1 + files_loc_dict[cik_year2_list[i]]).read()

        f.write(i_con.rstrip('\n'))
        f.write("\n")
        f.write(j_con.rstrip('\n'))
        f.write("\n")
    
        c = c + 1

    except:
        print("key not in dictionary!")

f.close()

print("count: ", c)
'''

'''
#uncomment this!
#doc2vec parameters
vector_size = 300
window_size = 15
min_count = 1
sampling_threshold = 1e-5
negative_size = 5
train_epoch = 100
dm = 0 #0 = dbow; 1 = dmpv
#dm = 1
worker_count = 120 #number of parallel processes

#pretrained word embeddings
#pretrained_emb = "/data/ftm/xgb_regr/doc2vec_toy/toy_data/pretrained_word_embeddings.txt"
pretrained_emb = "glove.6B.300d.txt"

#input corpus
train_corpus = open("train_docs_sec7.txt", 'r').readlines()

#output model
saved_path = "/data/ftm/xgb_regr/kdd_mlf_copy/sec7_glove_model.bin"

#enable logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

#train doc2vec model
docs = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(train_corpus)]

#Train doc2vec model
model = g.Doc2Vec(docs, size=vector_size, window=window_size, min_count=min_count, sample=sampling_threshold, workers=worker_count, hs=0, dm=dm, negative=negative_size, dbow_words=1, dm_concat=1, pretrained_emb=pretrained_emb, iter=train_epoch)

#save model
model.save(saved_path)
'''


#uncomment this!

df = pd.read_csv('text_similarity_scores.csv')

#find cosine similarity
model= Doc2Vec.load("sec7_glove_model.bin")

i = 0
j = 1

doc_sim = []

for i in range(0, 4204, 2): #Total number of document = 2102
    #cosine similarity
    doc_sim_score = cos_similarity(model.docvecs[i], model.docvecs[j])
    #Pearson correlation
    #doc_sim_score, _ = pearsonr(model.docvecs[i], model.docvecs[j])
    doc_sim.append(doc_sim_score)
    j = j + 2

df['d2v_glove_sim'] =  doc_sim
df.to_csv('text_similarity_scores.csv')

print("Total execution time: ", time.time() - start)