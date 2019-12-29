import numpy as np 
import tensorflow as tf 
from collections import defaultdict
from tensorflow.keras.models import Model 

def loss(y_true,y_pred):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y_pred))

def optimizer(learning_rate):
    return tf.train.GradientDescent(learning_rate)  

def create_model(vector_size,embedding_size):
    # Input Vectors 
    X = tf.placeholder(tf.float32,shape=[None,self.vec_len])
    Y = tf.placeholder(tf.float32,shape=[None,self.vec_len])

    #Dictionary of Weights
    weights = {
            'w1': tf.Variable(tf.random_normal([self.vec_len,self.EMBEDDING_SIZE])),
            'w2': tf.Variable(tf.random_normal([self.EMBEDDING_SIZE,self.vec_len])),
        } 

    #Dictionary of Biases
    biases = {
        'b1': tf.Variable(tf.random_normal([self.EMBEDDING_SIZE])),
        'b2': tf.Variable(tf.random_normal([self.vec_len])),
    }

    # Forward pass 
    hidden_1 = tf.add(tf.matmul(X,weights['w1']),biases['b1'])
    out_layer = tf.add(tf.matmul(hidden_1,weights['w2']),biases['b2'])
    softmax_out = tf.nn.softmax(out_layer)

    model = Model(inputs=[X,Y],outputs=[out_layer])
    emb = Model(inputs=[X],outputs=[softmax_out])
    return model,emb 



class word2vec:
    def __init__(self,corpus,window_size,embedding_size=5,learning_rate=0.01):
        self.corpus = corpus
        self.window_size = window_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
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

    def reset_model(self):
        self.model,self.emb_model = create_model(self.vec_len,self.embedding_size)
        opt = optimizer(learning_rate)
        self.model.compile(opt,loss)
        self.get_embeddings()

    def train(self,batch_size=5,epochs=10,initial_epochs=0,verbose=1):
        training_data = self.generate_training_data()
        if batch_size>len(training_data):
            batch_size = len(training_data)
        
        print(self.model.fit(training_data[1],training_data[0],batch_size=batch_size,epochs=epochs,
                                initial_epochs=initial_epochs,verbose=verbose,shuffle=False))
    
    def get_embeddings(self):
        pass 