import numpy as np
import tensorflow as tf
import sklearn.preprocessing as prep

import pandas as pd
import random as rd
import argparse


import itertools
from collections import Counter
import sys, re
import pandas as pd
import tensorflow as tf



#from tensorflow.examples.tutorials.mnist import input_data



#au_calss
def xavier_init(fan_in, fan_out, constant = 1):
    low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
    high = constant * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.random.uniform((fan_in, fan_out), minval = low, maxval = high, dtype= tf.float32)

class Autoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function = tf.nn.softplus,
                 optimizer = tf.compat.v1.train.AdamOptimizer(), scale = 0.0):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function
        self.scale = tf.compat.v1.placeholder(tf.float32)
        self.training_scale = scale
        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.x = tf.compat.v1.placeholder(tf.float32,[None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random.normal((n_input,)),
                                                     self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']),self.weights['b2'])
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)
        init = tf.compat.v1.global_variables_initializer()
        self.sess = tf.compat.v1.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden],dtype= tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype= tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype= tf.float32))
        return all_weights
    def partial_fit(self,X ):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict= {self.x: X,
                                                                           self.scale: self.training_scale})
        return cost
    def before_loss(self, X):
        cost = self.sess.run((self.cost), feed_dict={self.x: X,
                                                                          self.scale: self.training_scale})
        return cost
    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict= {self.x : X, self.scale: self.training_scale})
    def generate(self, hidden = None):
        if hidden is None:
            #print(self.weights["b1"].shape)
            hidden = np.random.normal( size = self.weights["b1"])
            #hidden = np.random.normal(size=self.weights["b1"].shape)


        return self.sess.run(self.reconstruction, feed_dict= {self.hidden: hidden})

    def reconstruct(self, X):

        return self.sess.run(self.reconstruction, feed_dict={self.x : X, self.scale: self.training_scale})
    def getWeights(self):
        return self.sess.run(self.weights['w1'])
    def getBias(self):
        return self.sess.run(self.weights['b1'])



#data_helpers



def get_samples(num_gene, args):

    disease_vector = pd.read_csv(args.input_disease) #205
    miro_vector = pd.read_csv(args.input_miRNA)  #244

    samples = []
    pos_file = args.input_positive
    neg_file = args.input_negative
    label = []
    disease = []
    micro = []
    with open(pos_file, "r") as f:
        for line in f:
            if line[0] ==' ':
                continue
            line_data = line.strip().split('\t')
            l = line_data[1]
            if line_data[0] in disease_vector:
                if line_data[1] in miro_vector:
                    samples.append((line_data[0],line_data[1]))
                    disease.append(line_data[0])
                    micro.append(line_data[1])
                    label.append([0, 1])



    with open(neg_file, "r") as f:
        for line in f:
            if line[0]==' ':
                continue
            line_data = line.strip().split('\t')
            if line_data[0] in disease_vector:
                if line_data[1] in miro_vector:
                    samples.append((line_data[0],line_data[1]))
                    disease.append(line_data[0])
                    micro.append(line_data[1])
                    label.append([1, 0])



    disease_vector = pd.read_csv(args.input_disease) #205
    miro_vector = pd.read_csv(args.input_miRNA)  #244
    vocab_size = len(samples)
   # df = pd.DataFrame(miro_vector)
    #print(df.shape[1])

    W = np.zeros(shape=(vocab_size, num_gene), dtype='float32')
    W[0] = np.zeros(num_gene, dtype='float32')
    i = 0
    for sample in samples:
        if sample[0] in disease_vector:
            if sample[1] in miro_vector:
                v1 = list(disease_vector[sample[0]])
                v2 = list(miro_vector[sample[1]])

                v1.extend(v2)
                W[i] = v1
                i = i + 1
    print(W[1])
    print(W[2])
    print(W[3])
    print(W[4])
    print(W[5])


    return W, label



def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    print(data.shape)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]




#AE_train



def parse_args():
    parser = argparse.ArgumentParser(description="Run autoencoder.")
    ## the input file
    ##disease-gene relationships and miRNA-gene relatiohships
    parser.add_argument('--input_disease', nargs='?', default='/Users/qiangdanlei/Desktop/graduate thesis/MDA-CNN-master/data/AE/disease_gene.csv',
                        help='Input disease_gene_relationship file')
    parser.add_argument('--input_miRNA', nargs='?', default='/Users/qiangdanlei/Desktop/graduate thesis/MDA-CNN-master/data/AE/miRNA_gene.csv',
                        help = 'Input miRNA_gene_relationship file')
    parser.add_argument('--input_positive',nargs = '?',default='/Users/qiangdanlei/Desktop/graduate thesis/MDA-CNN-master/data/AE/pos1.txt',
                        help='positive samples')
    parser.add_argument('--input_negative', nargs='?', default='/Users/qiangdanlei/Desktop/graduate thesis/MDA-CNN-master/data/AE/neg1.txt',
                        help='negative samples')
    parser.add_argument('--output', nargs='?', default='/Users/qiangdanlei/Desktop/graduate thesis/MDA-CNN-master/data/AE/result/disease_miRNA.csv',
                        help = 'Output low-dimensional disease_miRNA file')
    parser.add_argument('--label_file', nargs='?', default='/Users/qiangdanlei/Desktop/graduate thesis/MDA-CNN-master/data/AE/result/label.csv',
                        help='Output label file')
    parser.add_argument('--dimensions', nargs='?', default=1024,
                        help ='low dimensional representation')
    parser.add_argument('--batch_size', nargs='?', default=256,
                        help = 'number of samples in one batch')
    parser.add_argument('--training_epochs', nargs='?', default=100,
                        help= 'number of epochs in SGD')
    parser.add_argument('--display_step', nargs='?', default=10)
    parser.add_argument('--input_n_size', nargs='?', default=[3578, 2048])
    parser.add_argument('--hidden_size', nargs='?', default=[2048,1024])
    parser.add_argument('--gene_num', nargs= '?', default = 3578,
                        help= 'number of genes related to disease and miRNA')
    parser.add_argument('--transfer_function', nargs = '?', default= tf.nn.sigmoid,
                        help= 'the activation function')
    parser.add_argument('--optimizer', nargs='?',default= tf.compat.v1.train.AdamOptimizer,
                        help='optimizer for learning weights')
    parser.add_argument('--learning_rate',nargs= '?',default=0.001,
                        help='learning rate for the SGD')
    return parser.parse_args()

def standard_scale(X_train):
    # 对训练和测试数据进行标准化，需要注意的是必须保证训练集和测试集都使用完全相同的Scale
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    return X_train

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index: (start_index + batch_size)]

def main(args):
    gene_num = args.gene_num
    training_epochs = args.training_epochs
    batch_size = args.batch_size
    display_step = args.display_step
    input_n_size = args.input_n_size
    hidden_size = args.hidden_size
    transfer_function = args.transfer_function
    optimizer = args.optimizer
    learning_rate = args.learning_rate
    label_file = args.label_file
    data, label = get_samples(gene_num, args)
    label = pd.DataFrame(label)
    label.to_csv(label_file, header=None, index= None)
    sdne = []
    ###initialize
    for i in range(len(hidden_size)):
        print("i=",i)
        print("n_input" ,input_n_size[i],"n_hidden" ,hidden_size[i])
        ae = Autoencoder(n_input = input_n_size[i],n_hidden = hidden_size[i],
                            transfer_function = transfer_function,
                            optimizer = optimizer(learning_rate= learning_rate),
                            scale=0)
        sdne.append(ae)
    Hidden_feature = []
    for j in range(len(hidden_size)):
        if j == 0:
            X_train = standard_scale(data)
            print(X_train.shape)

        else:
            X_train_pre = X_train
            X_train = sdne[j-1].transform(X_train_pre)
            Hidden_feature.append(X_train)
            print(X_train.shape)

        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = int(X_train.shape[0] / batch_size)

            for batch in range(total_batch):

                batch_xs = get_random_block_from_data(X_train, batch_size)

                cost = sdne[j].partial_fit(batch_xs)
                #print("after = %f " % cost)

                avg_cost += cost / X_train.shape[0] * batch_size
                print(epoch)
            if epoch % display_step == 0:
                print("Epoch:", "%4d" % (epoch + 1), "cost:", "{:.9f}".format(avg_cost))


        if j == 0:
            feat0 = sdne[0].transform(standard_scale(data))
            data1 = pd.DataFrame(feat0)
            data1.T.to_csv(args.output)
        else:

            feat1 = sdne[j].transform(X_train)
            data1 = pd.DataFrame(feat1)
            data1.T.to_csv(args.output)

if __name__ == "__main__":
    args = parse_args()
    main(args)

