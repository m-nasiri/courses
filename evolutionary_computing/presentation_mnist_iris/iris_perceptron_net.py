import numpy as np
import tensorflow as tf
import scipy as sp
import random


iris_matfile_addr = 'iris.mat'
dataset = sp.io.loadmat(iris_matfile_addr)
samples = dataset['irisInputs'].T
labels = dataset['irisTargets'].T

data = list(zip(samples, labels))
random.shuffle(data)

samples = [x[0] for x in data]
samples = np.asarray(samples)
labels = [x[1] for x in data]
labels = np.asarray(labels)

samples_num = 150
train_num = 100
test_num = samples_num - train_num

samples_train = samples[0:train_num]
labels_train = labels[0:train_num]
samples_test = samples[train_num: ]
labels_test = labels[train_num: ]


class perceptron:
    def __init__(self, neuron_nums, n_classes):
        self.nc = n_classes
        self.nns = neuron_nums
        self.hlys_num = len(self.nns)
        self.x = tf.placeholder('float', [None, 4]) 
        self.y = tf.placeholder('float')
        
    def build_network(self):
           
        w = tf.Variable(tf.random_normal([4, self.nns[0]]))
        b = tf.Variable(tf.random_normal([self.nns[0]]))
        l = tf.add(tf.matmul(self.x, w) , b)
        l = tf.nn.relu(l)
        
        for lyr in range(1,self.hlys_num):
            w = tf.Variable(tf.random_normal([self.nns[lyr-1], self.nns[lyr]]))
            b = tf.Variable(tf.random_normal([self.nns[lyr]]))
            l = tf.add(tf.matmul(l, w) , b)
            l = tf.nn.relu(l)
        
        w = tf.Variable(tf.random_normal([self.nns[2], self.nc]))
        b = tf.Variable(tf.random_normal([self.nc]))
        self.logits = tf.matmul(l, w) + b
        

    def train_network(self):
        self.build_network()
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)
        
        hm_epochs = 1
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for itr in range(train_num):
                    epoch_x = samples_train[itr]
                    epoch_x = np.reshape(epoch_x, (-1, 4))
                    epoch_y = labels_train[itr]
                    _, c = sess.run([optimizer, cost], feed_dict= {self.x: epoch_x, self.y: epoch_y})
                    epoch_loss += c
                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            
            correct = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y,1))
            acc = tf.reduce_mean(tf.cast(correct, 'float'))
            self.accuracy = acc.eval({self.x:samples_test, self.y:labels_test})
            print('Accuracy = ', self.accuracy)




