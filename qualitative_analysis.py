import numpy as np
import pickle
import math
from numpy.linalg import norm
from itertools import chain
import matplotlib.pyplot as plt
 
from sklearn.manifold import TSNE

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

def get_closest_words(vec,vocab,U,V):
    close = np.Infinity
    near_idx = 0
    for i in range(len(vocab)):
        if(euclidean(get_embedding(i,U,V),vec) < close):
            close = euclidean(get_embedding(i,U,V),vec)
            near_idx = i
    return near_idx

def display_tsne_plot(U,V,word_set,filename):
    arr = np.empty((0,500),dtype=np.float32)
    for word in word_set:
        arr = np.append(arr,get_embedding(vocab.index(word),U,V).reshape(1,500),axis=0)
    tsne = TSNE(n_components=2,random_state=0,init='pca')
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_set, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.savefig(filename, format='png', dpi=150, bbox_inches='tight')
    plt.show()
    
with open("vocab.pkl","rb") as f:
    vocab = pickle.load(f)
with open("U.pkl","rb") as f:
    U = pickle.load(f)
with open("V.pkl","rb") as f:
    V = pickle.load(f)

with open('word2vec/trunk/questions-words.txt') as f:
    content = f.readlines()
content = [x.split() for x in content]

flat_content = list(chain.from_iterable(content))
word_set = list(set([word for word in flat_content if word in vocab]))

display_tsne_plot(U,V,word_set,'visual')
