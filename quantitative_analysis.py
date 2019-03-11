import numpy as np
import pickle
import math
from numpy.linalg import norm
from itertools import chain

def euclidean(vec1, vec2):
    diff = vec1 - vec2
    return math.sqrt(diff.dot(diff))

def cosine_sim(vec1, vec2):
    EPSILON = 1e-6
    vec1 += EPSILON * np.ones(len(vec1))
    vec2 += EPSILON * np.ones(len(vec1))
    return vec1.dot(vec2)/(norm(vec1)*norm(vec2))

def get_embedding(idx,U,V):
    return np.concatenate((U[idx,:],V[:,idx]))

def get_closest_indx(vec,vocab,U,V,exclude):
    close = np.Infinity
    near_idx = 0
    for i in range(len(vocab)):
        if(i in exclude):
            continue
        if(euclidean(get_embedding(i,U,V),vec) < close):
            close = euclidean(get_embedding(i,U,V),vec)
            near_idx = i
    return near_idx


with open("vocab.pkl","rb") as f:
    vocab = pickle.load(f)
with open("U.pkl","rb") as f:
    U = pickle.load(f)
with open("V.pkl","rb") as f:
    V = pickle.load(f)
    
with open('word2vec/trunk/questions-words.txt') as f:
    content = f.readlines()
content = [x.split() for x in content]

count = 0
acc = 0
for pairs in content:
    if(pairs[0] == ':'):
        continue
    else:
        if(pairs[0] in vocab and pairs[1] in vocab and pairs[2] in vocab and pairs[3] in vocab):
            count+=1
            w1 = vocab.index(pairs[0])
            w2 = vocab.index(pairs[1])
            w3 = vocab.index(pairs[2])    
            w4 = vocab.index(pairs[3])
            vec = get_embedding(w2,U,V) - get_embedding(w1,U,V) + get_embedding(w3,U,V)
            near_idx = get_closest_indx(vec,vocab,U,V,[w1,w2,w3])
            if(near_idx == w4):
                acc+=1
print(acc/count*100)