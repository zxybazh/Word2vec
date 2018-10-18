
# coding: utf-8

# In[1]:

import numpy as np
import random, math


def run(text_path='text8', vocab_path='vocab.txt', output_path='vectors.txt' )


    corpus = open(text_path, "r").read().split(" ")[1:]
    vocabulary = open(vocab_path, "r").read().split("\n")[:-1]



    # In[3]:

    dictionary ={}
    word_list = vocabulary + ["<unk>"]
    count = 0
    for w in word_list:
        dictionary[w] = count
        count += 1


    # In[4]:

    window_size = 10
    dimension = 100
    learning_rate = 0.1
    vocabulary_size = len(vocabulary) + 1
    x_max = 1000
    alpha = 0.75


    # In[5]:

    words = [-1 for i in xrange(len(corpus))]
    word_count = [0 for i in xrange(vocabulary_size)]
    i = 0
    for w in corpus:
        if w in dictionary:
            words[i] = dictionary[w]
        else:
            words[i] = vocabulary_size - 1
        word_count[words[i]] += 1
        i += 1


    # In[6]:

    cooccur = [[0 for i in xrange(vocabulary_size)] for j in xrange(vocabulary_size)]
    for i in xrange(len(words)):
        for y in words[i-window_size+1:i]:
            x = words[i]
            cooccur[x][y] += 1
            cooccur[y][x] += 1


    # In[7]:

    def f(x):
        if x > x_max: return 1
        else: return (float(x)/x_max)**alpha


    # In[8]:

    def compute_loss(W, b):
        loss = 0
        for i in xrange(vocabulary_size):
            for j in xrange(i+1, vocabulary_size):
                loss += f(cooccur[i][j])* (np.dot(W[i], W[j]) + b[i] + b[j] - np.log(1 + cooccur[i][j])) ** 2
        return loss


    # In[15]:

    W = (np.random.rand(vocabulary_size, dimension) - 0.5) / float(dimension + 1)
    b = (np.random.rand(vocabulary_size) - 0.5) / float(dimension + 1)


    # In[17]:

    for epoch in xrange(50):
        learning_rate = 0.1 / np.sqrt(1 + epoch)
        for i in xrange(len(words)):
            x = words[i]
            if (x == vocabulary_size -1): continue
            for y in words[i-random.randint(2, window_size)+1:i]:
                if (y == vocabulary_size - 1): continue
                r = np.dot(W[x], W[y]) + b[x] + b[y] - np.log(1 + cooccur[x][y])
                p = learning_rate * f(cooccur[x][y]) * r
                b[x] -= p
                b[y] -= p
                W_x_ = W[x].copy()
                W[x] -= p * W[y]
                W[y] -= p * W_x_

    # In[18]:

    file = open(output_path, "w")
    for i in xrange(4894):
        print >> file, word_list[i], " ".join([str(w) for w in W[i]])
    file.close()

