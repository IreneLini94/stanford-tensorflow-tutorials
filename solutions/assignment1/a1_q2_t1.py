import time
import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
MNIST = input_data.read_data_sets("/data/mnist",one_hot=True)

learning_rate = 1e-3
batch_size = 128
n_epoches = 15

X = tf.placeholder(tf.float32,[batch_size, 784])
Y = tf.placeholder(tf.float32,[batch_size, 10])
"""
w = tf.Variable(tf.random_normal(shape=[784,10], stddev=0.01), name="weights")
b = tf.Variable(tf.zeros([1, 10]), name="bias")
logits = tf.matmul(X,w) + b
"""
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.0001)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, padding):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding=padding)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1],
                         strides=[1,2,2,1],
                         padding='SAME')

X_image = tf.reshape(X,[-1,28,28,1])

W_conv1 = weight_variable([5,5,1,6])
b_conv1 = bias_variable([6])
conv1 = conv2d(X_image, W_conv1, "SAME") + b_conv1
pool1 = max_pool_2x2(conv1)

W_conv2 = weight_variable([5,5,6,16])
b_conv2 = bias_variable([16])
conv2 = conv2d(pool1, W_conv2, "VALID")
pool2 = max_pool_2x2(conv2)

W_conv3 = weight_variable([5,5,16,120])
b_conv2 = bias_variable([120])
conv3 = conv2d(pool2, W_conv3, "VALID")
pool3 = max_pool_2x2(conv3)

flat = tf.reshape(pool3,[-1,120])
W_fc1 = weight_variable([120, 84])
b_fc1 = bias_variable([84])
fc1 = tf.nn.relu(tf.matmul(flat, W_fc1) + b_fc1)
'''
keep_prob = tf.placeholder("float")
fc1_dropout = tf.nn.dropout(fc1, keep_prob)
'''
W_fc2 = weight_variable([84,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(fc1, W_fc2) + b_fc2)
loss = -tf.reduce_mean(Y*tf.log(y_conv))
#logits = tf.matmul(fc1, W_fc2) + b_fc2

#entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)
#loss = tf.reduce_mean(entropy)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    n_batches = int(MNIST.train.num_examples/batch_size)
    for i in range(n_epoches):
        for j in range(n_batches):
            X_batch, Y_batch = MNIST.train.next_batch(batch_size)
            _, batch_loss = sess.run([optimizer, loss], feed_dict={X:X_batch, Y:Y_batch})
            if (i*n_batches + j )%100 == 0:
                print "iter {} loss is : {}".format(i*n_batches+j, batch_loss)
    n_batches = int(MNIST.test.num_examples/batch_size)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = MNIST.test.next_batch(batch_size)
        _, loss_batch, preds = sess.run([optimizer, loss, y_conv],
                                                feed_dict={X:X_batch, Y:Y_batch})
        correct_preds = tf.equal(tf.argmax(preds,1), tf.argmax(Y_batch,1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds,tf.float32))
        total_correct_preds += sess.run(accuracy)
    print "Accuracy {0}".format(total_correct_preds/MNIST.test.num_examples)
