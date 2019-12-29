import numpy as np 
import tensorflow as tf 
from collections import defaultdict


class word2vec:
    def __init__(self,corpus,window_size):
        self.corpus = corpus
        self.window_size = window_size
        word_ = set()
        for sentence in self.corpus:
            for word in sentence:
                word_.add(word)
        self.word_list = list(word_)
        self.vec_len = len(self.word_list)
        self.word_idx = dict((word,i) for i,word in enumerate(self.word_list))
        self.idx_word = dict((i,word) for i,word in enumerate(self.word_list))
    
    def generate_training_data(self):
        training_data = []
        for sentence in self.corpus:
            for i,word in enumerate(sentence):
                target_word = self.onehotencoder(word)

                context_word = []
                for j in range(i-self.window_size,i+self.window_size):
                    if j>=0 and j<len(sentence) and j!=i:
                        context_word.append(self.onehotencoder(sentence[j]))
                training_data.append((context_word,target_word))
        return np.array(training_data)

    def onehotencoder(self,word):
        word_vec = [0 for i in range(self.vec_len)]
        word_idx = self.word_idx[word]
        word_vec[word_idx]=1
        return word_vec

    def create_model(self,):
        



