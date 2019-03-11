import math
import numpy
from operator import itemgetter
from numpy.linalg import norm
import sys
import os
import csv
import pickle

def euclidean(vec1, vec2):
    diff = vec1 - vec2
    return math.sqrt(diff.dot(diff))

def cosine_sim(vec1, vec2):
    EPSILON = 1e-6
    vec1 += EPSILON * numpy.ones(len(vec1))
    vec2 += EPSILON * numpy.ones(len(vec1))
    return vec1.dot(vec2)/(norm(vec1)*norm(vec2))

def assign_ranks(item_dict):
    ranked_dict = {}
    sorted_list = [(key, val) for (key, val) in sorted(item_dict.items(),
                                                     key=itemgetter(1),
                                                     reverse=True)]
    for i, (key, val) in enumerate(sorted_list):
        same_val_indices = []
        for j,(key2, val2) in enumerate(sorted_list):
            if val2 == val:
                same_val_indices.append(j+1)
        if len(same_val_indices) == 1:
            ranked_dict[key] = i+1
        else:
            ranked_dict[key] = 1.*sum(same_val_indices)/len(same_val_indices)
    return ranked_dict

def spearmans_rho(ranked_dict1, ranked_dict2):
    assert len(ranked_dict1) == len(ranked_dict2)
    if len(ranked_dict1) == 0 or len(ranked_dict2) == 0:
        return 0.
    x_avg = 1.*sum([val for val in ranked_dict1.values()])/len(ranked_dict1)
    y_avg = 1.*sum([val for val in ranked_dict2.values()])/len(ranked_dict2)
    num, d_x, d_y = (0., 0., 0.)
    for key in ranked_dict1.keys():
        xi = ranked_dict1[key]
        yi = ranked_dict2[key]
        num += (xi-x_avg)*(yi-y_avg)
        d_x += (xi-x_avg)**2
        d_y += (yi-y_avg)**2
    return num/(math.sqrt(d_x*d_y))

with open("SimLex-999/SimLex-999.txt","r") as f:
    word_pairs = []
    for line in csv.reader(f):
        word_pairs.append(line[0].split())


w1_idx = 0 
w2_idx = 1
simlex_idx = 3
USF_idx = 7

with open("vocab.pkl","rb") as f:
    vocab = pickle.load(f)
with open("U.pkl","rb") as f:
    U = pickle.load(f)
with open("V.pkl","rb") as f:
    V = pickle.load(f)

sim_dict = {}
word2vec_dict = {}
notfound = 0
total = 0
for i in range(len(word_pairs)):
    word1 = word_pairs[i][w1_idx]
    word2 = word_pairs[i][w2_idx]
    val = word_pairs[i][simlex_idx]
    if word1 in vocab and word2 in vocab:
        sim_dict[(word1,word2)] = float(val)
        word2vec_dict[(word1,word2)] =  cosine_sim(numpy.concatenate((V[:,vocab.index(word1)],U[vocab.index(word1),:])),numpy.concatenate((V[:,vocab.index(word2)],U[vocab.index(word2),:])))
    else:
        notfound += 1
    total += 1
print("total:",total)
print("notfound:",notfound)
print("Spearman_coeffient:",spearmans_rho(assign_ranks(word2vec_dict),assign_ranks(sim_dict)))
