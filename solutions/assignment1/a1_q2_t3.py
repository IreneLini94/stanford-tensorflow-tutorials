#encoding:utf-8
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

#读取数据
rawdata = pd.read_csv("../../data/heart.csv")

#最后一列是预测目标，将其进行onehot encode
Y = rawdata.pop("chd")
Y = Y.apply(str)
Y = pd.get_dummies(Y,prefix="chd")

#对数据进行一些预处理，新增两个对年龄和sbp进行分段的字段。
age_bins = [0,18,30,45,60,100]
rawdata["age_group"] = pd.cut(rawdata["age"], age_bins, labels=["a1","a2","a3","a4","a5"])
'''
typea_bins = [10,47,53,60,80]
rawdata["typea_group"]  = pd.cut(rawdata["typea"], typea_bins, labels=["t1","t2","t3","t4"])
'''
sbp_bins = [100,120,130,150,200,250]
rawdata["sbp_group"]  = pd.cut(rawdata["sbp"], sbp_bins, labels=["s1","s2","s3","s4","s5"])
#对类别数据进行one_hot encode 并进行归一化
rawdata = pd.get_dummies(rawdata)
N = RobustScaler()
scale_data = N.fit_transform(rawdata)

#划分为训练数据和测试数据
train_X, test_X, train_Y, test_Y = train_test_split(scale_data,Y.values, test_size=0.1, random_state=1)

#设置超参数
learning_rate = 1e-3
batch_size = 32
n_epoches = 120

X = tf.placeholder(tf.float32,[None, train_X.shape[1]])
Y = tf.placeholder(tf.float32,[None, 2])
"""
w1 = tf.Variable(tf.random_normal(shape=[10,24], stddev=0.0001), name="weights1")
b1 = tf.Variable(tf.zeros([1, 24]), name="bias1")
h1 = tf.nn.relu(tf.matmul(X,w1) + b1)

w2 = tf.Variable(tf.random_normal(shape=[24,2], stddev=0.0001), name = "weights2")
b2 = tf.Variable(tf.zeros([1,2]),name="bias2")
logits = tf.nn.relu(tf.matmul(h1,w2)+b2)
"""
#设计简单的一层的神经网络分类器
w = tf.Variable(tf.random_normal(shape=[train_X.shape[1],2], stddev=0.0001), name="weights")
b = tf.Variable(tf.zeros([1, 2]), name="bias")
logits = tf.matmul(X,w) + b

#计算准确率
correct_preds = tf.equal(tf.argmax(logits,1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(correct_preds,tf.float32))

#设置目标函数
entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)
loss = tf.reduce_mean(entropy)

#设置优化方式
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

init = tf.global_variables_initializer()

#开始训练
with tf.Session() as sess:
    sess.run(init)
    num_examples = train_X.shape[0]
    n_batches = int(num_examples/batch_size)
    for i in range(n_epoches):
        for j in range(n_batches):
            start = batch_size*j
            end = batch_size*(j+1)
            X_batch = train_X[start:end]
            Y_batch = train_Y[start:end]
            if (i*n_batches + j )%10 == 0:
                batch_loss, batch_accuracy = sess.run([loss, accuracy],feed_dict={X:X_batch,Y:Y_batch})
                print "iter {} loss is : {:.5f} and accuracy is :{:5f}".format(i*n_batches+j, batch_loss, batch_accuracy)
            sess.run(optimizer, feed_dict={X:X_batch, Y:Y_batch})
    #在测试集上进行测试
    print "Test Accuracy is {0}".format(accuracy.eval(feed_dict={X:test_X,Y:test_Y}))

#按照这个方案准确率在70%周围波动，而且波动范围比较大，这个效果并不是很理想。
#尝试用sklearn的lr模型跑也差不多是这个结果
#可以尝试继续调参和特征工程应该可以做的更好一些。