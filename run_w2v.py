
# coding: utf-8

# In[1]:

import numpy as np
import random, math
from nltk import *

def run(text_path='text8', vocab_path='vocab.txt', output_path='vectors.txt' )


    # In[4]:

    freq = FreqDist(corpus).most_common(10000)
    for w in freq:
        if not w[0] in vocabulary:
            vocabulary.append(w[0])


    # In[5]:

    window_size = 5
    dimension = 100
    learning_rate = 0.1
    vocabulary_size = len(vocabulary) + 1
    negtive_sample_size = 5


    # In[6]:

    dictionary ={}
    word_list = vocabulary + ["<unk>"]
    count = 0
    for w in word_list:
        dictionary[w] = count
        count += 1


    # In[7]:

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


    # In[8]:

    norm_rate = 0.75
    sum_cover = []
    for i, count in enumerate(word_count[:-1]):
        x = count**norm_rate
        if (i == 0): sum_cover = [x]
        else: sum_cover.append(sum_cover[-1]+x)


    # In[9]:

    def sigmoid(x):
        if (x > 10): return 1.0
        if (x <-10): return 0.0
        return 1.0 / (1.0 + math.exp(-x))


    # In[10]:

    def find_word_by_cover(c):
        l = 1
        r = vocabulary_size-2
        ans = r-2
        while (l <= r):
            mid = (l + r) / 2
            if (sum_cover[mid] >= c):
                ans = mid
                r = mid - 1
            else:
                l = mid + 1
        return ans


    # In[11]:

    def get_negative(pos_samples):
        global words, sum_cover
        neg_samples = []
        temp = negtive_sample_size
        for j in xrange(temp):
            x = find_word_by_cover(np.random.rand()*sum_cover[-1])
            if not (x in pos_samples): neg_samples.append((x, 0))
        return neg_samples


    # In[12]:

    W1 = np.random.uniform(low=-0.5/dimension, high=0.5/dimension, size=(vocabulary_size, dimension))
    W2 = np.zeros((vocabulary_size, dimension))
    delta = np.zeros(dimension)


    # In[15]:

    progress = 0


    # In[16]:

    for epoch in xrange(10):
        word_shown = [0 for i in xrange(vocabulary_size)]
        learning_rate = 0.1/np.sqrt(epoch+1)
        for i in xrange(len(words)):
            k = words[i]
            progress = i+1
            word_shown[k] += 1
            if (k == vocabulary_size -1): continue
            loss = 0
            pos = words[i-random.randint(1, window_size)+1 : i] + words[i+1 : i+random.randint(1, window_size)]

            for w in pos:
                if w == vocabulary_size -1: continue
                delta.fill(0)
                samples = [(w, 1)] + get_negative(pos)
                for token, label in samples:
                    y = np.dot(W1[k], W2[token])
                    p = sigmoid(y)
                    loss += (-2*label + 1) * np.log(p+1)
                    g = -learning_rate * (p - label)
                    delta += g * W2[token]
                    W2[token] += g * W1[k]
                W1[k] += delta
    # In[17]:
    #save_embedding(word_list, W1):
    file = open(output_path, "w")
    for i in xrange(4894):
        print >> file, word_list[i], " ".join([str(w) for w in W1[i]])
    file.close()


    # In[ ]:



