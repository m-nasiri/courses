import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# address the folder contain MNIST dataset files
mnist_dataset_addr = "/mnt/Document/AI/Dataset/MNIST"
mnist = input_data.read_data_sets(mnist_dataset_addr, one_hot=True)


class perceptron:
    def __init__(self, neuron_nums, n_classes):
        self.nc = n_classes
        self.nns = neuron_nums
        self.hlys_num = len(self.nns)
        self.x = tf.placeholder('float', [None, 784]) 
        self.y = tf.placeholder('float')
        
    def build_network(self):
           
        w = tf.Variable(tf.random_normal([784, self.nns[0]]))
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
        
        batch_size = 100
        hm_epochs = 100
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            
            for epoch in range(hm_epochs):
                epoch_loss = 0
                for _ in range(int(mnist.train.num_examples/batch_size)):
                    epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                    _, c = sess.run([optimizer, cost], feed_dict= {self.x: epoch_x, self.y: epoch_y})
                    epoch_loss += c
                print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
            
            correct = tf.equal(tf.argmax(self.logits,1), tf.argmax(self.y,1))
            acc = tf.reduce_mean(tf.cast(correct, 'float'))
            self.accuracy = acc.eval({self.x:mnist.test.images, self.y:mnist.test.labels})


        



