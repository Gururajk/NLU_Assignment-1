#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 07:52:47 2019

@author: gururaj
"""

import numpy as np
from nltk.corpus import reuters
from nltk import word_tokenize
import re
from nltk.corpus import stopwords
import nltk
import math
from random import choices
from random import sample
import pickle
import time


def tokenize(text):
    min_length = 3
    cachedStopWords = stopwords.words("english")
    words = map(lambda word: word.lower(), word_tokenize(text));
    tokens = [word for word in words
                  if word not in cachedStopWords]
    p = re.compile('[a-zA-Z]+');
    filtered_tokens =list(filter(lambda token: p.match(token) and len(token)>=min_length,tokens));
    return filtered_tokens

def calc_unigram_dist(bigdoc,vocab):
    fdist = nltk.FreqDist(bigdoc)
    palpha = 0.75

    pdist = np.zeros(len(vocab))
    norm = 0
    for i in vocab:
        norm = norm + math.pow(fdist[i],palpha)
    for i in range(len(vocab)):
        pdist[i] = math.pow(fdist[vocab[i]],palpha) / norm
    return pdist

def getsamples(token,k,vocab,pdist):
    samples = [values for values in choices(vocab,pdist,k=k) if values != token]
    return samples

def sigmoid(z):
    return 1/(1+math.exp(-z))

def main():
    documents = reuters.fileids()

    docs = [reuters.raw(doc_id) for doc_id in documents]
    
    tokenized_docs = [tokenize(doc) for doc in docs]
    
    bigdoc = [y for x in tokenized_docs for y in x]
    
    vocab = list(set(bigdoc))

    with open("vocab.pkl","wb") as f:
        pickle.dump(vocab,f)

    indxd_tokenized_docs = [[vocab.index(y) for y in x] for x in tokenized_docs]
    
    pdist = calc_unigram_dist(bigdoc,vocab)

    neg_smpl_words = [getsamples(token,5000,list(range(len(vocab))),pdist) for token in list(range(len(vocab)))]
    
    learning_rate = 0.01
    epochs = 75

    n = 250
    m = 2
    k = 5

#    V = np.random.randn(n,len(vocab))
#    U = np.random.randn(len(vocab),n)
    with open("V.pkl","rb") as f:
        V = pickle.load(f)
    with open("U.pkl","rb") as f:
        U = pickle.load(f)

    for epoach in range(epochs):
        start = time.time()
        err_epch = 0
        for idx_doc in indxd_tokenized_docs:
            examples = []
            for indx in range(len(idx_doc)):
                token_idx = idx_doc[indx]
                context_words_idx = idx_doc[max(indx-m,0):indx] + idx_doc[indx+1:min(indx+m+1,len(idx_doc))]
                for cntx_word_idx in context_words_idx:
                    examples.append([token_idx,[(cntx_word_idx,1)] + [(negsmpl_idx,0) for negsmpl_idx in sample(neg_smpl_words[token_idx],k)]])
            err_tk = 0
            for (token_idx,exmp) in examples:
                err_ctx = 0
                cgv = np.zeros(n)
                v = V[:,token_idx]
                for (target,label) in exmp:
                    u = U[target,:]
                    z = np.dot(u,v)
                    p = sigmoid(z)
                    err_ctx = err_ctx - np.log(sigmoid(z * (2*label - 1)))
                    g = learning_rate * (label - p)
                    cgv = cgv + g * u
                    U[target,:] = u + g * v
                V[:,token_idx] = v + cgv
                err_tk = err_tk + err_ctx/(k+1)
            err_epch = err_epch + err_tk/(len(examples)+1)
        print("error :",err_epch,file=open("log.txt","a"))
        with open("V.pkl","wb") as f:
            pickle.dump(V,f)
        with open("U.pkl","wb") as f:
            pickle.dump(U,f)
        end = time.time()
        print("time :",end - start,file=open("log.txt","a"))
    

if __name__ == "__main__":
    main()
